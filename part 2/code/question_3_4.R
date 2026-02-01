# ----------------------------- 1. LOADING DATA -----------------------------------
library(tidyverse)
library(janitor)
library(boot)
library(broom)
library(patchwork)
library(sandwich)
library(rsample)
library(caret)
library(randomForest)

# Setting the seed
set.seed(1)

# Loading the data
df_train_file <- read_csv("~/Desktop/Classes/Data Science/project 1/cleaned_data/cleaned_data.csv")
df_test_file  <- read_csv("~/Desktop/Classes/Data Science/project 1/cleaned_data/test_data.csv")

# Defining full data
data_full <- bind_rows(df_train_file, df_test_file)

# Factoring categoricals
categorical_cols <- c(
  "overdue_policy",
  "interlibrary_relation_code",
  "fscs_definition_code",
  "locale_code",
  "beac_code"
)

# Factoring categoricals
for (v in categorical_cols) {
  data_full[[v]] <- factor(data_full[[v]])
}

# Defining data size
n <- nrow(data_full)

# Creating train-test split 50-50
index <- sample(seq_len(n), size = floor(n / 2))
df_train <- data_full[index, ]
df_test  <- data_full[-index, ]

# -----------------------------3(a) Fitting Models ------------------------------

# Defining our model
formula_obj <- as.formula(
  "log(visits) ~ log(population_lsa) +
   log1p(print_volumes) +
   log1p(ebook_volumes) +
   log(county_population) + 
   num_bookmobiles +
   (num_lib_branches) +
   overdue_policy +
   interlibrary_relation_code +
   fscs_definition_code +
   locale_code +
   beac_code"
)

# Training our model
model_train <- lm(formula_obj, data = df_train)

# Getting summary
summary(model_train)

# Getting estimates
tidy_train <- broom::tidy(model_train) %>%
  clean_names() %>%
  rename(
    estimate = estimate,
    std_error = std_error,
    t_value = statistic,
    pr_t = p_value
  ) %>%
  mutate(significant_train = pr_t < 0.05)

# Plotting A1 - A4
res_train <- broom::augment(model_train, df_train)

p1 <- ggplot(res_train, aes(.fitted, .resid)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_smooth(se = FALSE) +
  labs(title = "Residuals vs Fitted — TRAIN",
       x = "Fitted values", y = "Residuals") +
  theme_minimal()

p2 <- ggplot(res_train, aes(.fitted, sqrt(abs(.std.resid)))) +
  geom_point(alpha = 0.5) +
  geom_smooth(se = FALSE) +
  labs(title = "Scale–Location — TRAIN",
       x = "Fitted values", 
       y = "sqrt(|Standardized Residual|)") +
  theme_minimal()

p3 <- ggplot(res_train, aes(x = .resid)) +
  geom_histogram(aes(y = ..density..), bins = 100,
                 fill = "grey70", color = "black") +
  stat_function(fun = dnorm,
                args = list(mean = mean(res_train$.resid, na.rm = TRUE),
                            sd = sd(res_train$.resid, na.rm = TRUE)),
                linewidth = 1.2) +
  labs(title = "Residual Histogram — TRAIN",
       x = "Residuals", y = "Density") +
  theme_minimal()

(p1 | p2) / p3

# -----------------------3(b) Fitting Models on Test ------------------------------

# Fitting data on test
model_test <- lm(formula_obj, data = df_test)
summary(model_test)

# Defining test estimates
tidy_test <- broom::tidy(model_test) %>%
  clean_names() %>%
  rename(
    estimate = estimate,
    std_error = std_error,
    t_value = statistic,
    pr_t = p_value
  ) %>%
  mutate(significant_test = pr_t < 0.05)

# Fitting plots on test
res_test <- broom::augment(model_test, df_test)

ggplot(res_test, aes(.fitted, .resid)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_smooth(se = FALSE) +
  labs(title = "Residuals vs Fitted — TEST", x = "Fitted values", y = "Residuals") +
  theme_minimal()

ggplot(res_test, aes(.fitted, sqrt(abs(.std.resid)))) +
  geom_point(alpha = 0.5) +
  geom_smooth(se = FALSE) +
  labs(title = "Scale–Location — TEST",
       x = "Fitted values", y = "sqrt(|Standardized Residual|)") +
  theme_minimal()

ggplot(res_test, aes(x = .resid)) +
  geom_histogram(aes(y = ..density..), bins = 100, fill = "grey70", color = "black") +
  stat_function(fun = dnorm,
                args = list(mean = mean(res_test$.resid, na.rm = TRUE),
                            sd = sd(res_test$.resid, na.rm = TRUE)),
                linewidth = 1.2) +
  labs(title = "Residual Histogram — TEST", x = "Residuals", y = "Density") +
  theme_minimal()

# Defining a comparison between train and test
comparison <- tidy_train %>%
  dplyr::select(estimate_train = estimate, term, train_p = pr_t, significant_train) %>%
  left_join(
    tidy_test %>% dplyr::select(estimate_test = estimate, term, test_p = pr_t, significant_test),
    by = "term"
  )

comparison

# -----------------------3(c) Bootstrap ------------------------------
# Prepping data
df_boot_base <- df_train %>%
  mutate(
    log_visits = log(visits),
    log_pop = log(population_lsa),
    log_print = log1p(print_volumes),
    log_ebook = log1p(ebook_volumes),
    log_county = log(county_population)
  ) %>%
  ungroup()

# Defining model
boot_formula <- log_visits ~ log_pop + log_print + log_ebook + log_county +
  num_bookmobiles + num_lib_branches +
  overdue_policy + interlibrary_relation_code +
  fscs_definition_code + locale_code + beac_code

# Fitting basic model
model_boot_ref <- lm(boot_formula, data = df_boot_base)
coef_names <- names(coef(model_boot_ref))

# Setting seed
set.seed(1)

# Defining bootstraps
boots <- bootstraps(df_boot_base, times = 2000)

# Fitting model in each bootstrap
boot_coefs <- boots %>%
  mutate(
    model = map(splits, ~ try(lm(boot_formula, data = analysis(.x)), silent = TRUE)),
    tidy  = map(model, function(m) {
      if (inherits(m, "try-error")) {
        tibble(term = coef_names, estimate = NA_real_)
      } else {
        est <- coef(m)
        tibble(term = coef_names, estimate = est[coef_names])
      }
    })
  ) %>%
  select(id, tidy) %>%
  unnest(tidy)

# Getting BS summary
boot_summary <- boot_coefs %>%
  group_by(term) %>%
  summarize(
    boot_se   = sd(estimate, na.rm = TRUE),
    perc_low  = quantile(estimate, 0.025, na.rm = TRUE),
    perc_high = quantile(estimate, 0.975, na.rm = TRUE)
  )

standard_ci <- confint(model_boot_ref)
standard_se <- coef(summary(model_boot_ref))[, "Std. Error"]
standard_results <- tibble(
  term = rownames(standard_ci),
  estimate = coef(model_boot_ref),
  std_se = standard_se,
  std_low = standard_ci[, 1],
  std_high = standard_ci[, 2]
)

# Defining comparison 
comparison <- standard_results %>%
  left_join(boot_summary, by = "term") %>%
  mutate(
    sig_standard  = (std_low  > 0 | std_high < 0),
    sig_bootstrap = (perc_low > 0 | perc_high < 0),
    changed_significance = sig_standard != sig_bootstrap,
    width = std_se < boot_se
  )

comparison

# -----------------------3(d) Correction ------------------------------

p_adjusted_bonferroni <- p.adjust(tidy_train$pr_t, method = "bonferroni")
p_adjusted_bh <- p.adjust(tidy_train$pr_t, method = "BH")

adjusted_results <- tidy_train %>%
  mutate(
    p_adj_bonferroni = p_adjusted_bonferroni,
    p_adj_bh = p_adjusted_bh
  ) %>%
  select(term, p_adj_bonferroni, p_adj_bh)

tidy_train %>%
  left_join(adjusted_results, by = "term") %>%
  mutate(
    sig_raw  = pr_t < 0.05,
    sig_bonf = p_adj_bonferroni < 0.05,
    sig_bh   = p_adj_bh < 0.05
  )

# -----------------------4(b) AIPW ------------------------------
set.seed(1)

# Defining dataframe
df <- df_train %>%
  mutate(
    overdue_bin = ifelse(overdue_policy == levels(overdue_policy)[2], "Yes", "No"),
    overdue_bin = factor(overdue_bin, levels = c("No", "Yes")),
    log_visits = log(visits),
    log_pop_lsa = log(population_lsa),
    log_print = log1p(print_volumes),
    log_ebook = log1p(ebook_volumes),
    log_county_pop = log(county_population),
    across(
      c(interlibrary_relation_code, fscs_definition_code,
        locale_code, beac_code),
      as.factor
    ),
    T = as.numeric(overdue_bin == "Yes")
  )

# Defining variables
vars <- c(
  "log_pop_lsa", "log_print", "log_ebook", "log_county_pop",
  "num_bookmobiles", "num_lib_branches",
  "interlibrary_relation_code", "fscs_definition_code",
  "locale_code", "beac_code"
)

# Defining PS
formula_ps <- reformulate(vars, "overdue_bin")
form_outcome <- reformulate(vars, "log_visits")

# Defining k for cross-val
K <- 5
folds <- createFolds(df$T, k = K, list = TRUE, returnTrain = FALSE)

df$ps_cf  <- NA
df$mu1_cf <- NA
df$mu0_cf <- NA

# Doing internal cross val for AIPW
for (k in 1:K) {
  
  test_idx  <- folds[[k]]
  train_idx <- setdiff(seq_len(nrow(df)), test_idx)
  
  df_train_k <- df[train_idx, ]
  df_test_k  <- df[test_idx, ]
  
  ctrl_k <- trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  )
  
  rf_ps_k <- train(
    formula_ps,
    data = df_train_k,
    method = "rf",
    metric = "Accuracy",
    trControl = ctrl_k,
    tuneLength = 5
  )
  
  ps_pred <- predict(rf_ps_k, newdata = df_test_k, type = "prob")[, "Yes"]
  ps_pred <- pmin(pmax(ps_pred, 0.01), 0.99)
  
  out_mod_1_k <- randomForest(
    form_outcome,
    data = df_train_k %>% filter(T == 1),
    ntree = 100
  )
  
  out_mod_0_k <- randomForest(
    form_outcome,
    data = df_train_k %>% filter(T == 0),
    ntree = 100
  )
  
  mu1_pred <- predict(out_mod_1_k, newdata = df_test_k)
  mu0_pred <- predict(out_mod_0_k, newdata = df_test_k)
  
  df$ps_cf[test_idx]  <- ps_pred
  df$mu1_cf[test_idx] <- mu1_pred
  df$mu0_cf[test_idx] <- mu0_pred
}

df <- df %>%
  mutate(
    aipw_term1 = T * (log_visits - mu1_cf) / ps_cf,
    aipw_term0 = (1 - T) * (log_visits - mu0_cf) / (1 - ps_cf),
    Gamma_hat  = mu1_cf - mu0_cf + aipw_term1 - aipw_term0
  )

AIPW_ATE_CF <- mean(df$Gamma_hat)
AIPW_SE <- sd(df$Gamma_hat) / sqrt(nrow(df))

ci_lower <- AIPW_ATE_CF - 1.96 * AIPW_SE
ci_upper <- AIPW_ATE_CF + 1.96 * AIPW_SE

results_list <- list(
  AIPW_ATE = AIPW_ATE_CF,
  AIPW_SE  = AIPW_SE,
  CI_95    = c(ci_lower, ci_upper),
  predictions = df %>% select(log_visits, ps_cf, mu1_cf, mu0_cf, Gamma_hat)
)

print(results_list)
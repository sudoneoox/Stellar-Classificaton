# =============================================================================
# SDSS17 Stellar Classification - Support Vector Machine
# =============================================================================

library(e1071)
library(caret)

# ---- 0. File Paths -----------------------------------------------------------
DATA_DIR  <- file.path("C:/Users/diego/Downloads/Stellar-Classificaton", "data")
df_train  <- readRDS(file.path(DATA_DIR, "df_supervised_train.rds"))
df_test   <- readRDS(file.path(DATA_DIR, "df_supervised_test.rds"))
df_clean  <- readRDS(file.path(DATA_DIR, "df_clean.rds"))

# Separate features and labels
train_features <- df_train %>% select(-class)
train_labels   <- as.factor(df_train$class)
test_features  <- df_test %>% select(-class)
test_labels    <- as.factor(df_test$class)

cat("=== DATA LOADED ===\n")
cat("Training:", nrow(df_train), "rows\n")
cat("Testing: ", nrow(df_test), "rows\n\n")

# ---- 1. Subsample for Tuning ------------------------------------------------
# SVM on 80k rows is computationally expensive. We subsample 15k rows
# for kernel comparison and hyperparameter tuning, then train the final
# model on the full training set with optimal parameters.
set.seed(42)
tune_idx     <- sample(1:nrow(df_train), size = 15000)
tune_data    <- df_train[tune_idx, ]
tune_features <- tune_data %>% select(-class)
tune_labels   <- as.factor(tune_data$class)

cat("=== TUNING SUBSAMPLE ===\n")
cat("Subsample size:", nrow(tune_data), "\n")
cat("Class distribution:\n")
print(table(tune_labels))
cat("\n")

# ---- 2. Kernel Comparison ----------------------------------------------------
# Compare linear, polynomial, and RBF kernels with default parameters
cat("=== KERNEL COMPARISON (default params, tuning subsample) ===\n")

kernels <- c("linear", "polynomial", "radial")
kernel_results <- data.frame(kernel = character(), accuracy = numeric(),
                             time_sec = numeric(), stringsAsFactors = FALSE)

for (k in kernels) {
  cat("Training kernel:", k, "... ")
  t_start <- Sys.time()
  
  model <- svm(x = tune_features, y = tune_labels, kernel = k,
               cost = 1, gamma = 1 / ncol(tune_features))
  
  t_end <- Sys.time()
  elapsed <- as.numeric(difftime(t_end, t_start, units = "secs"))
  
  preds <- predict(model, tune_features)
  acc   <- mean(preds == tune_labels)
  
  cat("Accuracy:", round(acc, 4), "| Time:", round(elapsed, 1), "sec\n")
  kernel_results <- rbind(kernel_results,
                          data.frame(kernel = k, accuracy = acc,
                                     time_sec = round(elapsed, 1)))
}

cat("\nKernel comparison summary:\n")
print(kernel_results)
best_kernel <- kernel_results$kernel[which.max(kernel_results$accuracy)]
cat("\nBest kernel:", best_kernel, "\n\n")

# ---- 3. Coarse Grid Search (RBF) --------------------------------------------
# Tune cost and gamma using 5-fold CV on the tuning subsample.
# Using 5-fold instead of 10-fold to keep runtime manageable.
cat("=== COARSE GRID SEARCH ===\n")
cat("Kernel:", best_kernel, "\n")
cat("Grid: cost = {0.1, 1, 10, 100} x gamma = {0.01, 0.1, 0.5, 1}\n")
cat("Method: 5-fold CV on 15k subsample\n")
cat("This may take several minutes...\n\n")

t_start <- Sys.time()

tune_data$class <- as.factor(tune_data$class)

coarse_tune <- tune(
  svm,
  class ~ .,
  data = tune_data,
  kernel = best_kernel,
  ranges = list(
    cost  = c(0.1, 1, 10, 100),
    gamma = c(0.01, 0.1, 0.5, 1)
  ),
  tunecontrol = tune.control(cross = 5)
)

t_end <- Sys.time()
cat("Coarse grid search completed in",
    round(as.numeric(difftime(t_end, t_start, units = "mins")), 1), "minutes\n\n")

cat("=== COARSE GRID RESULTS ===\n")
print(coarse_tune$performances)
cat("\nBest parameters (coarse):\n")
print(coarse_tune$best.parameters)
cat("Best CV error:", round(coarse_tune$best.performance, 4), "\n\n")

# ---- 4. Train Final Model on Full Training Set ------------------------------
# Use the best parameters from tuning to train on all 80k training rows.
optimal_cost  <- coarse_tune$best.parameters$cost
optimal_gamma <- coarse_tune$best.parameters$gamma

cat("=== TRAINING FINAL MODEL ===\n")
cat("Kernel:", best_kernel, "\n")
cat("Cost:", optimal_cost, "\n")
cat("Gamma:", optimal_gamma, "\n")
cat("Training on full training set (", nrow(df_train), " rows)...\n")

t_start <- Sys.time()

svm_final <- svm(x = train_features, y = train_labels,
                 kernel = best_kernel,
                 cost = optimal_cost, gamma = optimal_gamma)

t_end <- Sys.time()
cat("Training completed in",
    round(as.numeric(difftime(t_end, t_start, units = "mins")), 1), "minutes\n\n")

cat("=== FINAL MODEL SUMMARY ===\n")
print(summary(svm_final))

# ---- 5. Test Set Evaluation --------------------------------------------------
cat("=== TEST SET EVALUATION ===\n")
test_preds <- predict(svm_final, test_features)

# Confusion matrix
cat("Confusion Matrix:\n")
cm <- table(Predicted = test_preds, Actual = test_labels)
print(cm)

# Overall test error
test_error <- mean(test_preds != test_labels)
test_acc   <- 1 - test_error
cat("\nTest Error Rate:", round(test_error, 4), "\n")
cat("Test Accuracy:  ", round(test_acc, 4), "\n\n")

# Per-class metrics
cat("Per-Class Accuracy:\n")
for (cls in levels(test_labels)) {
  cls_idx <- test_labels == cls
  cls_acc <- mean(test_preds[cls_idx] == test_labels[cls_idx])
  cat("  ", cls, ":", round(cls_acc, 4), "\n")
}
cat("\n")

# ---- 6. Refit on Full Dataset ------------------------------------------------
# Project requirement: fit best model on the full dataset (no train/test split)
# and provide model summary, support vector counts, and boundary plots.
cat("=== REFIT ON FULL DATASET ===\n")

# Scale the full clean dataset the same way
full_features <- df_clean %>% select(-class)
full_features_scaled <- as.data.frame(scale(full_features))
full_labels <- as.factor(df_clean$class)

cat("Full dataset:", nrow(df_clean), "rows\n")
cat("Refitting with cost =", optimal_cost, ", gamma =", optimal_gamma, "\n")

t_start <- Sys.time()

svm_full <- svm(x = full_features_scaled, y = full_labels,
                kernel = best_kernel,
                cost = optimal_cost, gamma = optimal_gamma)

t_end <- Sys.time()
cat("Refit completed in",
    round(as.numeric(difftime(t_end, t_start, units = "mins")), 1), "minutes\n\n")

cat("=== FULL MODEL SUMMARY ===\n")
print(summary(svm_full))

# Support vectors per class
cat("\nSupport Vectors per Class:\n")
print(svm_full$nSV)
cat("Total Support Vectors:", sum(svm_full$nSV), "\n")
cat("Proportion of data as SVs:",
    round(sum(svm_full$nSV) / nrow(df_clean), 4), "\n\n")

# ---- 7. Decision Boundary Plots ---------------------------------------------
# Plot SVM boundary for selected feature pairs.
# Using a subsample for plotting speed.
cat("=== GENERATING BOUNDARY PLOTS ===\n")

set.seed(42)
plot_idx <- sample(1:nrow(full_features_scaled), size = 5000)
plot_features <- full_features_scaled[plot_idx, ]
plot_labels   <- full_labels[plot_idx]

# Pair 1: redshift vs r (strongest class separator + a magnitude filter)
cat("Plotting: redshift vs r\n")
svm_plot1 <- svm(class ~ redshift + r, data = cbind(plot_features, class = plot_labels),
                 kernel = best_kernel, cost = optimal_cost, gamma = optimal_gamma)
plot(svm_plot1, cbind(plot_features, class = plot_labels),
     redshift ~ r, main = "SVM Boundary: redshift vs r")

# Pair 2: redshift vs u (redshift vs another filter)
cat("Plotting: redshift vs u\n")
svm_plot2 <- svm(class ~ redshift + u, data = cbind(plot_features, class = plot_labels),
                 kernel = best_kernel, cost = optimal_cost, gamma = optimal_gamma)
plot(svm_plot2, cbind(plot_features, class = plot_labels),
     redshift ~ u, main = "SVM Boundary: redshift vs u")

# Pair 3: r vs i (two correlated magnitude filters)
cat("Plotting: r vs i\n")
svm_plot3 <- svm(class ~ r + i, data = cbind(plot_features, class = plot_labels),
                 kernel = best_kernel, cost = optimal_cost, gamma = optimal_gamma)
plot(svm_plot3, cbind(plot_features, class = plot_labels),
     r ~ i, main = "SVM Boundary: r vs i")

cat("\n=== SVM PIPELINE COMPLETE ===\n")
cat("Save these for 07_comparison.R:\n")
cat("  Test error rate:", round(test_error, 4), "\n")
cat("  Test accuracy:  ", round(test_acc, 4), "\n")
cat("  Best kernel:    ", best_kernel, "\n")
cat("  Best cost:      ", optimal_cost, "\n")
cat("  Best gamma:     ", optimal_gamma, "\n")
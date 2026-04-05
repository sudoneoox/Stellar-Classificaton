# =============================================================================
# SDSS17 Stellar Classification - Data Preprocessing Pipeline
# =============================================================================

# Change this to your project dir
DATASET = file.path("C:/Users/diego/Downloads/Stellar-Classificaton", "data", "star_classification.csv")
PROJECT_DIR = "C:/Users/diego/Downloads/Stellar-Classificaton"


library(tidyverse)

# ---- 1. Load Raw Data -------------------------------------------------------
df <- read.csv(DATASET)
cat("=== RAW DATA ===\n")
cat("Dimensions:", dim(df), "\n\n")

# ---- 2. Drop Metadata Columns -----------------------------------------------
metadata_cols <- c("obj_ID", "run_ID", "rerun_ID", "cam_col",
                   "field_ID", "spec_obj_ID", "plate", "MJD", "fiber_ID")

df_clean <- df %>% select(-all_of(metadata_cols))
cat("=== AFTER DROPPING METADATA ===\n")
cat("Dropped:", paste(metadata_cols, collapse = ", "), "\n")
cat("Remaining columns:", paste(names(df_clean), collapse = ", "), "\n\n")

# ---- 3. Remove Sentinel Values ----------------------------------------------
# SDSS uses -9999 as a placeholder for missing/unreliable photometry.
# Only 1 row is affected (all three of u, g, z are -9999 in same row).
before <- nrow(df_clean)
df_clean <- df_clean %>% filter(u != -9999 & g != -9999 & z != -9999)
after <- nrow(df_clean)
cat("=== SENTINEL REMOVAL ===\n")
cat("Rows removed:", before - after, "\n")
cat("Remaining rows:", after, "\n\n")

# ---- 4. Drop Positional Features --------------------------------------------
df_clean <- df_clean %>% select(-alpha, -delta)
cat("=== AFTER DROPPING POSITIONAL FEATURES ===\n")
cat("Dropped: alpha, delta\n")
cat("Remaining columns:", paste(names(df_clean), collapse = ", "), "\n")
cat("Dimensions:", dim(df_clean), "\n\n")

# ---- 5. Save Cleaned (Unscaled) Data ----------------------------------------
saveRDS(df_clean, file.path(PROJECT_DIR, "data", "df_clean.rds"))
cat("Saved: data/df_clean.rds\n\n")

# ---- 6. Separate Features and Labels ----------------------------------------
class_labels <- df_clean$class
features <- df_clean %>% select(-class)

cat("=== FEATURE SUMMARY (PRE-SCALING) ===\n")
print(summary(features))
cat("\n")

# ---- 7. Scale Features ------------------------------------------------------
features_scaled <- as.data.frame(scale(features))

cat("=== FEATURE SUMMARY (POST-SCALING) ===\n")
cat("Means (should be ~0):\n")
print(round(colMeans(features_scaled), 6))
cat("\nSDs (should be ~1):\n")
print(round(apply(features_scaled, 2, sd), 6))
cat("\n")

# ---- 8. Supervised Learning: Train/Test Split --------------------------------
# 80/20 split as required by the project.
# set.seed ensures KNN and SVM use identical splits.
set.seed(42)
n <- nrow(features_scaled)
train_idx <- sample(1:n, size = floor(0.8 * n))
test_idx <- setdiff(1:n, train_idx)

# Training set
train_features <- features_scaled[train_idx, ]
train_labels <- class_labels[train_idx]
df_train <- train_features
df_train$class <- train_labels

# Testing set
test_features <- features_scaled[test_idx, ]
test_labels <- class_labels[test_idx]
df_test <- test_features
df_test$class <- test_labels

cat("=== SUPERVISED SPLIT ===\n")
cat("Training set:", nrow(df_train), "rows\n")
cat("Testing set: ", nrow(df_test), "rows\n")
cat("\nTraining class distribution:\n")
print(table(df_train$class))
cat("\nTesting class distribution:\n")
print(table(df_test$class))
cat("\n")

# Save supervised datasets
saveRDS(df_train, file.path(PROJECT_DIR,"data", "df_supervised_train.rds"))
saveRDS(df_test, file.path(PROJECT_DIR,"data", "df_supervised_test.rds"))
cat("Saved: data/df_supervised_train.rds\n")
cat("Saved: data/df_supervised_test.rds\n\n")

# ---- 9. Unsupervised Learning: Full Scaled Data (No Labels) -----------------
# Clustering operates on the full dataset without class labels.
# Labels are saved separately for post-hoc alignment checks.
df_unsupervised <- features_scaled

saveRDS(df_unsupervised, file.path(PROJECT_DIR,"data", "df_unsupervised.rds"))
saveRDS(class_labels, file.path(PROJECT_DIR,"data", "class_labels.rds"))
cat("=== UNSUPERVISED DATA ===\n")
cat("Dimensions:", dim(df_unsupervised), "\n")
cat("Saved: data/df_unsupervised.rds\n")
cat("Saved: data/class_labels.rds\n\n")

# ---- 10. Summary of All Saved Files -----------------------------------------
cat("=============================================================\n")
cat("PREPROCESSING COMPLETE - OUTPUT FILES:\n")
cat("=============================================================\n")
cat("df_clean.rds            -> Cleaned, unscaled, with class\n")
cat("data", "df_supervised_train.rds -> Scaled, with class, 80% split\n")
cat("df_supervised_test.rds  -> Scaled, with class, 20% split\n")
cat("df_unsupervised.rds     -> Scaled, no class labels\n")
cat("class_labels.rds        -> Class labels for alignment checks\n")
cat("=============================================================\n")
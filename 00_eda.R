install.packages(c("tidyverse", "corrplot", "factoextra", "cluster", "GGally", "gridExtra", "e1071"))

library(tidyverse)
library(corrplot)
library(factoextra)
library(cluster)
library(GGally)
library(gridExtra)
library(e1071)

# Change this to your dataset file
DATASET = file.path("C:/Users/diego/Downloads/Stellar-Classificaton", "data", "star_classification.csv")


df <- read.csv(DATASET)

# Basic structure
cat("=== DIMENSIONS ===\n")
print(dim(df))

cat("\n=== COLUMN TYPES ===\n")
print(str(df))

cat("\n=== FIRST 6 ROWS ===\n")
print(head(df))

cat("\n=== SUMMARY STATISTICS ===\n")
print(summary(df))

cat("=== CLASS DISTRIBUTION ===\n")
print(table(df$class))
print(prop.table(table(df$class)))

ggplot(df, aes(x = class, fill = class)) +
  geom_bar() +
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5) +
  labs(title = "Class Distribution", x = "Object Class", y = "Count") +
  theme_minimal()

cat("=== MISSING VALUES PER COLUMN ===\n")
print(colSums(is.na(df)))

cat("\n=== TOTAL MISSING ===\n")
cat(sum(is.na(df)), "\n")

cat("\n=== DUPLICATE ROWS ===\n")
cat(sum(duplicated(df)), "\n")

# Identify columns to drop
metadata_cols <- c("obj_ID", "run_ID", "rerun_ID", "cam_col", 
                   "field_ID", "spec_obj_ID", "plate", "MJD", "fiber_ID")

# Keep only meaningful features + class
df_clean <- df %>% select(-all_of(metadata_cols))

cat("=== REMAINING COLUMNS ===\n")
print(names(df_clean))
cat("Dimensions:", dim(df_clean), "\n")

# Boxplots for each numeric feature grouped by class
numeric_cols <- df_clean %>% select(-class) %>% names()

plots <- lapply(numeric_cols, function(col) {
  ggplot(df_clean, aes(x = class, y = .data[[col]], fill = class)) +
    geom_boxplot(outlier.size = 0.3) +
    labs(title = col, x = "", y = col) +
    theme_minimal() +
    theme(legend.position = "none")
})

do.call(grid.arrange, c(plots, ncol = 3))


# Correlation among numeric features
cor_matrix <- cor(df_clean %>% select(-class))

cat("=== CORRELATION MATRIX ===\n")
print(round(cor_matrix, 3))

corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45,
         addCoef.col = "black", number.cex = 0.7,
         title = "Feature Correlation Matrix",
         mar = c(0, 0, 2, 0))

cat("=== SKEWNESS PER FEATURE ===\n")
skew_vals <- sapply(df_clean %>% select(-class), skewness)
print(round(skew_vals, 3))


# Density plots overlaid by class for the photometric filters + redshift
key_features <- c("u", "g", "r", "i", "z", "redshift")

plots2 <- lapply(key_features, function(col) {
  ggplot(df_clean, aes(x = .data[[col]], fill = class)) +
    geom_density(alpha = 0.4) +
    labs(title = col, x = col) +
    theme_minimal()
})

do.call(grid.arrange, c(plots2, ncol = 2))

# Sample down for plotting speed
set.seed(42)
df_sample <- df_clean %>% sample_n(2000)

ggpairs(df_sample, columns = c("u", "g", "r", "i", "z", "redshift"),
        aes(color = class, alpha = 0.4),
        upper = list(continuous = wrap("cor", size = 3)),
        lower = list(continuous = wrap("points", size = 0.5))) +
  theme_minimal()
library(ggplot2)

data <- read.csv('public_v2_042618.csv')

# 1) Feature selection
# ====================
# Try to get down to about 15 features (before dummies)
# Use 3 methods: chi-square, GINI, random forest

# Chi-square

# GINI

# Random Forest

# 2) Cross validation set up
# ==========================
# Use 10-fold cross validation and validation set

# Shuffle data

# Separate validation set (stratify!)

# Split 10 folds (stratify!)

# 3) SVM
# ======
# For every kernel of SVM
# Apply CV10
# Impute missing values
# Save results to table

# Print table values
# See which SVM kernel is best

# 4) Make artificial data set
# ===========================
# Create "aritificial" data set with SVM

# 5) Random forests
# =================
# Ver 1 (control): uses original data set
# Ver 2: random forest with artificial data

# Ver 1

# Ver 2

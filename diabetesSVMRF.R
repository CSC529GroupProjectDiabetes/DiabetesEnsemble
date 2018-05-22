library(ggplot2)
library(data.table)
library(caret)

data <- read.csv('public_v2_042618.csv')

# Seed number for random number generator: seed
seed <- 1234

# Label name: label
# (Note: 1 = is diabetic, 2 = not diabetic)
label <- "DBTS_NEW"

# Drop first column, since it's the index
data <- data[-c(1)]
# Print first 10 rows to compare with shuffled data
head(data)

# Change non-numeric columns to factors
# Note: not sure about some of the columns I marked as factor should be a
# factor; please contact me to discuss this
numericCol <- c("CAPI_WT", "EXAM_WT", "ACASI_WT", "BLOOD_WT", "SPAGE",
                "HSQ_2", "HSQ_3", "HSQ_4", "HSQ_5", "OHQ_4", "OHQ_5",
                "ALQ_1","BMI")
data[!(names(data) %in% numericCol)] <- lapply(data[!(names(data) %in% numericCol)], factor)

# Check if above code worked
sapply(data, class)

# Be sure that DBTS_NEW class 1 is the "true" variable
data$DBTS_NEW <- factor(data$DBTS_NEW, levels = c("1", "2"))

# Get index of dependent variable: yindex
yindex <- grep(label, colnames(data))

# Get the list of colnames: tmpcolname
tmpcolname <- names(data)

# 1) Cross validation set up
# ==========================
# Set up overview:
# - Shuffle data
# - Prepare 10-fold CV (no holdout, since small sample)

# Shuffle data
set.seed(seed)
data <- data[sample(1:nrow(data)), ]
# Print first 10 rows of shuffled data
head(data)

# Show how many people are diabetic
summary(data$DBTS_NEW)

# Split k = 10 folds (stratify!)
set.seed(seed)
folds <- createFolds(factor(data$DBTS_NEW), k = 10, list = FALSE)
data$fold <- folds

# 2) Feature selection
# ====================
# Try to get down to about 15 features (before dummies)
# Use 3 methods: chi-square, GINI, random forest

# Chi-square
# For each variable
# If the variable number isn't yindex
# Get the p.value
# For every feature available
# If the colname isn't

# GINI

# Random Forest


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

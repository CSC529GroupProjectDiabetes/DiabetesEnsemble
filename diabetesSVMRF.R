library(ggplot2)
library(caret)
library(e1071)
library(missForest)
library(randomForest)
library(MLmetrics)
library(dummy)

ori <- read.csv('public_v2_042618.csv')

# Seed number for random number generator: seed
seed <- 1234

# Label name: label
# (Note: 1 = is diabetic, 2 = not diabetic)
label <- "DBTS_NEW"

# Drop first column, since it's the index
ori <- ori[-c(1)]
data <- ori
# Print first 10 rows to compare with shuffled data
head(data)

# Change non-numeric columns to factors
# Note: not sure about some of the columns I marked as factor should be a
# factor; please contact me to discuss this
numericCol <- c("CAPI_WT", "EXAM_WT", "ACASI_WT", "BLOOD_WT", "SPAGE",
                "HSQ_2", "HSQ_3", "HSQ_4", "HSQ_5", "OHQ_4", "OHQ_5",
                "DBQ_2", "ALQ_1","BMI", "EDU4CAT", "HSQ_1", "PAQ_1",
                "DBQ_3UNIT", "DBQ_4UNIT", "DBQ_5UNIT", "DBQ_6UNIT",
                "DBQ_8UNIT", "POVGROUP4_0812", "SSQ_1", "SSQ_2", "SSQ_3",
                "SSQ_4", "SSQ_5", "SSQ_6")
data[!(names(data) %in% numericCol)] <- lapply(data[!(names(data) %in% numericCol)], factor)

# While we're at it, specify which columns are ordinal/categorical
# Going to use ordinal as numeric
# ordinalCol <- c("EDU4CAT", "HSQ_1", "PAQ_1", "DBQ_3UNIT", "DBQ_4UNIT",
#                 "DBQ_5UNIT", "DBQ_6UNIT", "DBQ_8UNIT", "POVGROUP4_0812",
#                 "SSQ_1", "SSQ_2", "SSQ_3", "SSQ_4", "SSQ_5", "SSQ_6")
factorCol <- c("MARITAL", "BORN", "DMQ_12", "RACE", "HIQ_1", "HUQ_3",
               "HUQ_14", "OHQ_1", "OHQ_2", "OHQ_3", "BPQ_2", "MCQ_17",
               "PAQ_2", "PAQ_7", "PAQ_8", "PAQ_11", "PAQ_14", "PAQ_16",
               "PAQ_17", "PAG2008", "DBQ_2", "SMQ_1", "SMQ_12", "SMQ_14",
               "SMOKER4CAT", "INQ_3")

# Check if above code worked
sapply(data, class)

# Be sure that DBTS_NEW class 1 is the "true" variable
data$DBTS_NEW <- factor(data$DBTS_NEW, levels = c("1", "2"))

# Get index of dependent variable: yindex
yindex <- grep(label, colnames(data))

# Get the list of colnames: tmpcolname
tmpcolname <- names(data)

# 1) Training/testing set up
# ==========================
# Set up overview:
# - Shuffle data
# - Prepare 10-fold CV (no holdout, since small sample)

# Shuffle data
set.seed(seed)
data <- data[sample(1:nrow(data)), ]
# Print first 10 rows of shuffled data
head(data)
 
# # Show how many people are diabetic
# summary(data$DBTS_NEW)
# 
# # Split k = 10 folds (stratify!)
# set.seed(seed)
# folds <- createFolds(factor(data$DBTS_NEW), k = 10, list = FALSE)
# data$fold <- folds

# If we want to use separate testing set for the end, use this code:
set.seed(seed)
trainInd <- createDataPartition(factor(data$DBTS_NEW), p = 0.8, list = FALSE)
train <- data[trainInd, ]
test <- data[-trainInd, ]

# 2) Missing data imputation/data normalization
# =============================================
# Do imputation and normalization on entire data (since imputing and
# normalizing for each fold is kind of a hassle)

# Keeps min/max of numeric columns: minMax
minMax <- function(x){
  xmin <- min(x, na.rm = T)
  xmax <- max(x, na.rm = T)
  return(c(xmin, xmax))
}

# Data frame containing min/max of each column without NA's: trNumParam
# Note: row 1 is min, row 2 is max
trNumParam <- data.frame(apply(train[, numericCol], MARGIN = 2, FUN = minMax))

# Applies 0-1 normalization to a column: colNorm
# - x: column of interest
# - p: associated min and max of columns
colNorm <- function(x, p){
  tMin <- p[1,]
  tMax <- p[2,]
  return((x - tMin)/(tMax - tMin))
}

# Do 0-1 normaliztions for all dataframe columns: dfNorm
# - df: dataframe to be normalized
# - param: min/max of all columns to be normalized
# - ncols: list of numeric columns
dfnorm <- function(df, paramdf, numcols){
  for(n in 1:length(numcols)){
    df[, numcols[n]] <- colNorm(df[numcols[n]], paramdf[numcols[n]])
  }
  return(df)
}

# Normalize/impute relevant columns: datamanip
# * The returned data is a "missForest" data type. To access the imputed
#   dataframe, use varname$ximp
datamanip <- function(df, paramdf, numcols){
  tmpdf <- dfnorm(df, paramdf, numcols)
  dfImp <- missForest(tmpdf, variablewise = T)
  print(dfImp$OOBerror)
  return(dfImp)
}

trainImp <- datamanip(train, trNumParam, numericCol)

normTrain <- trainImp$ximp

# 3) Feature selection
# ====================

# Split k = 10 folds
set.seed(seed)
folds <- createFolds(factor(normTrain$DBTS_NEW), k = 10, list = FALSE)
normTrain$folds <- folds

# Try to get down to about 10 features (before dummies)
# In Han paper, used 3 methods: chi-square, GINI, random forest

# Trying random forest for now?

# List of variables to keep: selectFeat
selectFeat <- c()

# The next two variables are for when we denormalize the data later
# List of numeric columns after feature extraction: newNumC
newNumC <- c()

# Dataframe containing only min/max of newNumC: newNumP
newNumP <- trNumParam[, newNumC]

# df with only selected features: selectTrain
selectTrain <- normTrain[, selectFeat]
selectTrain <- normTrain[, label]

# Generate dummy variables
selectTrain <- dummy.data.frame(selectTrain)

# 4) SVM
# ======
# Try to find best set of support vectors and apply to entire training
# data for random forest; use e1071 grid search

# NOTE: Using code by statcompute on R-bloggers as basis
# ======================================================
# <start>
pkgs <- c('foreach', 'doParallel')
lapply(pkgs, require, character.only = T)
registerDoParallel(cores = 4)

# All feature-selected variables as string
xvar <- paste()
fml <- as.formula(paste("as.factor(DBTS_NEW) ~ ", xvar))

# Hyperparameter ranges to be tested
cost <- c(10, 100)
gamma <- c(1, 2)
parms <- expand.grid(cost = cost, gamma = gamma)

### LOOP THROUGH PARAMETER VALUES ###
result <- foreach(i = 1:nrow(parms), .combine = rbind) %do% {
  c <- parms[i, ]$cost
  g <- parms[i, ]$gamma
  ### K-FOLD VALIDATION ###
  out <- foreach(j = 1:max(selectTrain$folds), .combine = rbind, .inorder = FALSE) %dopar% {
    deve <- selectTrain[selectTrain$folds != j, ]
    test <- selectTrain[selectTrain$folds == j, ]
    mdl <- e1071::svm(fml, data = deve, type = "C-classification", kernel = "radial",
                      cost = c, gamma = g, probability = TRUE)
    # Adding a bit of my code to get number of SV's
    svcount <- length(mdl$SV)
    pred <- predict(mdl, test, decision.values = TRUE, probability = TRUE)
    data.frame(y = test$DBTS_NEW, prob = attributes(pred)$probabilities[, 2],
               sv = svcount)
  }
  ### CALCULATE SVM PERFORMANCE ###
  roc <- pROC::roc(as.factor(out$y), out$prob) 
  data.frame(parms[i, ], roc = roc$auc[1], out$sv)
}
# <end>

# Print table values
result

# See which SVM hyperparameter is best with plot

# Save best hyperparameters: svmC, svmG
svmC <- NULL
svmG <- NULL

# 5) Make artificial data set
# ===========================
# Create "artificial" data set with best SVM hyperparameter

# Generate the model with optimal SVM hyperparameter: svmMdl
svmMdl <- e1071::svm(fml, data = selectTrain, type = "C-classification",
                     kernel = "radial", cost = svmC, gamma = svmG,
                     probability = TRUE)

# Predict labels: predSVM
predSVM <- predict(svmMdl, selectTrain, decision.values = TRUE,
                   probability = TRUE)

# Assign feature columns to new dataframe: artificial
artificial <- selectTrain[, selectFeat]

# Function to denormalize all numeric variables: deNorm
deNorm <- function(x, p){
  tMin <- p[1,]
  tMax <- p[2,]
  return(x * (tMax - tMin) + tMin)
}

# Denormalize all numeric columns: dfNorm
# - df: dataframe to be denormalized
# - param: min/max of all columns to be denormalized
# - ncols: list of numeric columns
dfDnorm <- function(df, paramdf, numcols){
  for(n in 1:length(numcols)){
    df[, numcols[n]] <- deNorm(df[numcols[n]], paramdf[numcols[n]])
  }
  return(df)
}

# Denormalize all normalized variables
artificial[, newNumC] <- dfDnorm(artificial[, newNumC], newNumP, newNumC)

# Assign predicted labels as new label in artificial data set
artificial$DBTS_NEW <- predSVM

# 6) Random forests
# =================
# Using "artificial" data, do random forest using grid search

# Used stackOverflow answer as reference:
customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree", "nodesize"),
                                  class = rep("numeric", 3),
                                  label = c("mtry", "ntree", "nodesize"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, mtry = param$mtry, ntree=param$ntree,
               nodesize=param$nodesize, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

# This accuracy measure calculates precision, recall, and F1 scores: metric
metric <- "prSummary"

conrol <- trainControl(method = "repeatedcv", number = 10, repeats = 3,
                       search = "random")
set.seed(seed)

# Choose hyperparameter ranges accordingly
tunegrid <- expand.grid(mtry = c(1:15), ntree = c(1:15),
                        nodesize = c(1:15))
set.seed(seed)
rf_gridsearch <- train(label~., data = artificial, method = "customRF",
                       metric = metric, tuneGrid = tunegrid,
                       trControl = control)

# Choose best hyperparameter
plot(rf_gridsearch)

# Best random forest hyperparameter: rfmtry, rfntree, rfnodeS

# Compare predicted results with original training set lables
set.seed(seed)
rfMdl <- randomForest(fml, data = artificial, mtry = rfmtry,
                      ntree = rfntree, nodesize = rfnodeS)
predrf <- predict(rfMdl)

# Confusion matrix of SVM + RF training model: rfcm
traincm <- confusionMatrix(predrf, selectTrain$DBTS_NEW)

# 7) Apply model to test set
# ==========================

# Save min/max of test set: tsNumParam
tsNumParam <- data.frame(apply(test[, numericCol], MARGIN = 2, FUN = minMax))

# Normalize/impute test set
testImp <- datamanip(test, tsNumParam, numericCol)
normTrain <- trainImp$ximp

# Keep needed features


# Predict using SVM model from training

# Denormalize

# Predict using random forest model from training


# 8) Compare with logistic regression
# ===================================
# Since logit is commonly used in many diabetes diagnosis data, use the
# same feature as 

# Use normalized data from training in 10-fold CV for logit

# Predict using logit parameters

# Compare with SVM+RF model

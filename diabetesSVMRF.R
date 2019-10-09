library(ggplot2)
library(caret)
library(e1071)
library(missForest)
library(randomForest)
library(MLmetrics)
library(dummies)
library(plyr)

# setwd("C:/Users/Ryoh/Documents/CSC529/CSC529GroupProject")

ori <- read.csv('public_v2_042618.csv')

# Seed number for random number generator: seed
seed <- 1234

# 1) Data preparation based on Dan's feature selection
# ====================================================

# Drop first column, since it's the index
ori <- ori[-c(1)]

# Create new data frame so that original df isn't touched: df
df <- ori

# Print first 10 rows to compare with shuffled data
head(df)

# List of variables to keep: selectFeat
selectFeat <- c("SPAGE", "MCQ_17", "INQ_3", "HSQ_1", "EDU4CAT", "MARITAL",
                "BPQ_2", "HUQ_14", "ALQ_1", "OHQ_5", "DBTS_NEW")

df <- ori[, colnames(ori) %in% selectFeat]

# EDA for some numeric columns
ggplot(df, aes(ALQ_1)) + geom_histogram(binwidth = 12) # Once a month or less
ggplot(df, aes(SPAGE)) + geom_histogram(binwidth = 10)

# Based on histogram of ALQ_1, convert this to a boolean var
df$ALQ_1 <- df$ALQ_1 <= 12

# Variables to use as categorical variables: factorvar
factorvar <- c("EDU4CAT", "MARITAL", "HUQ_14", "BPQ_2", "DBTS_NEW",
               "MCQ_17", "INQ_3", "ALQ_1")
numericCol <- c("SPAGE", "HSQ_1", "OHQ_5")

# Convert all categorical variables to factors
df[, factorvar] <- lapply(df[, factorvar], as.factor)

# Check if above code worked
sapply(df, class)

# Change name of factor variables so that it doesn't interfere with
# random forest later on
df$DBTS_NEW <- revalue(df$DBTS_NEW, c("1" = "diabetic",
                                      "2" = "nondiabetic"))

# 2) Training/testing set up
# ==========================

# Shuffle data
set.seed(seed)
df <- df[sample(1:nrow(df)), ]
# Print first 10 rows of shuffled data
head(df)
 
# Show how many people are diabetic
summary(df$DBTS_NEW)

# Split training-testing (80:20):
set.seed(seed)
trainInd <- createDataPartition(factor(df$DBTS_NEW), p = 0.8, list = FALSE)
train <- df[trainInd, ]
test <- df[-trainInd, ]

# 3) Missing data imputation/data normalization
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
trNumParam <- data.frame(apply(train[, numericCol],
                               MARGIN = 2, FUN = minMax))

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
  set.seed(seed)
  tmpdf <- dfnorm(df, paramdf, numcols)
  dfImp <- missForest(tmpdf, variablewise = T)
  print(dfImp$OOBerror)
  return(dfImp)
}

trainImp <- datamanip(train, trNumParam, numericCol)

normTrain <- trainImp$ximp
normTcopy <- trainImp$ximp

# Apply dummy variables for marital status
normTrain <- dummy.data.frame(normTrain, names = "MARITAL")


# 4) SVM
# ======
# Try to find best set of support vectors and apply to entire training
# data for random forest; use e1071 grid search

set.seed(seed)
svmControl <- tune.control(random = T, nrepeat = 10, sampling = "cross",
                           cross = 10)

set.seed(seed)
svmTune <- tune(svm, DBTS_NEW~., data = normTrain,
                ranges = list(gamma = 2 ^ (-5:1),
                              cost = 2 ^ (-3:5)),
                tunecontrol = svmControl, scale = F)

summary(svmTune)
plot(svmTune)

# Get perfomrance plot later

# Save best hyperparameters: svmC, svmG
svmC <- 0.5
svmG <- 0.5

# 5) Make artificial data set
# ===========================
# Create "artificial" data set with best SVM hyperparameter

# Generate the model with optimal SVM hyperparameter: svmMdl
svmMdl <- e1071::svm(DBTS_NEW~., data = normTrain, type = "C-classification",
                     kernel = "radial", cost = svmC, gamma = svmG,
                     probability = TRUE, scale = F)

predSVM <- predict(svmMdl, normTrain, decision.values = TRUE,
                   probability = TRUE)
SVMcm <- confusionMatrix(predSVM, train$DBTS_NEW)

# Assign feature columns to new dataframe: artificial
artificial <- normTcopy

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
artificial[, numericCol] <- dfDnorm(artificial[, numericCol], trNumParam, numericCol)

# <For assignment 3>
# hw3 <- artificial
# hw3$DBTS_NEW <- selectTrain$DBTS_NEW
# write.csv(hw3, file = "nychanesHW3.csv", row.names = F)

artificial$DBTS_NEW <- predSVM

# 6) Random forests
# =================
# Using "artificial" data, do random forest using grid search

control <- trainControl(method="repeatedcv", number=10, repeats=10, search="grid",
                        summaryFunction = prSummary, classProbs = T)

# Tuning ntree
tunegrid <- expand.grid(.mtry=c(sqrt(ncol(artificial))))
modellist <- list()
for (ntree in c(5, 10, 15, 20, 25)) {
  set.seed(seed)
  fit <- train(DBTS_NEW ~ ., data=artificial, method="rf", metric = "AUC",
               tuneGrid=tunegrid, trControl=control, ntree=ntree)
  key <- toString(ntree)
  modellist[[key]] <- fit
}
# compare results
results <- resamples(modellist)
summary(results)
dotplot(results)

# Tuning nodesize with best ntree
rbftree <- 15
modellist <- list()
for (ns in c(1, 3, 5, 7, 9)){
  set.seed(seed)
  fit <- train(DBTS_NEW ~ ., data=artificial, method="rf", metric = "AUC",
               tuneGrid=tunegrid, trControl=control, ntree=rbftree, nodesize = ns)
  key <- toString(ns)
  modellist[[key]] <- fit
}
results <- resamples(modellist)
summary(results)
dotplot(results)

# Tuning mtry with best nodesize and ntree
rbfnode <- 1
# rbfnode <- 3
modellist <- list()
set.seed(seed)
tunegrid <- expand.grid(.mtry=c(1:10))
for(m in c(1:10)){
  fit <- train(DBTS_NEW~., data = artificial,
               method="rf", metric = "AUC", tuneGrid=tunegrid,
               nodesize = rbfnode, ntree = rbftree,
               trControl=control, .mtry = m)
  key <- toString(m)
  modellist[[key]] <- fit
}
results <- resamples(modellist)
summary(results)
dotplot(results)

# Best random forest hyperparameter: rfmtry, rfntree, rfnodeS
rfmtry <- 4

# Compare predicted results with original training set lables
set.seed(seed)
rfMdl <- randomForest(DBTS_NEW~., data = artificial,
                      ntree = rbftree, nodesize = rbfnode, mtry = rfmtry)
predrf <- predict(rfMdl, artificial)

# Confusion matrix of SVM + RF training model: rfcm
traincm <- confusionMatrix(predrf, normTrain$DBTS_NEW)

# Calculate f1 score: F1
F1 <- function(prec, rec){
  return(2 * prec * rec / (prec + rec))
}

# 7) Apply model to test set
# ==========================

# Save min/max of test set: tsNumParam
tsNumParam <- data.frame(apply(test[, numericCol], MARGIN = 2, FUN = minMax))

# Normalize/impute/denormalize test set
set.seed(seed)
testImp <- datamanip(test, tsNumParam, numericCol)

normTest <- testImp$ximp

normTest[, numericCol] <- dfDnorm(normTest[, numericCol], tsNumParam, numericCol)

# Predict with rf model
svmrfpred <- predict(rfMdl, normTest)
finalcm <- confusionMatrix(svmrfpred, normTest$DBTS_NEW)

# 8) Try a plain-ole' random forest
# =================================

oriImp <- artificial[, -c(8)]
oriImp$DBTS_NEW <- normTrain$DBTS_NEW

# Tuning ntree
tunegrid <- expand.grid(.mtry=c(sqrt(ncol(oriImp))))
modellist <- list()
for (ntree in c(5, 10, 15, 20, 25, 30)) {
  set.seed(seed)
  fit <- train(DBTS_NEW ~ ., data=oriImp, method="rf", metric = "AUC",
               tuneGrid=tunegrid, trControl=control, ntree=ntree)
  key <- toString(ntree)
  modellist[[key]] <- fit
}
# compare results
results <- resamples(modellist)
summary(results)
dotplot(results)

# Tuning nodesize with best ntree
tree <- 20
modellist <- list()
for (ns in c(1, 3, 5, 7, 9)){
  set.seed(seed)
  fit <- train(DBTS_NEW ~ ., data=oriImp, method="rf", metric = "AUC",
               tuneGrid=tunegrid, trControl=control, ntree=tree, nodesize = ns)
  key <- toString(ns)
  modellist[[key]] <- fit
}
results <- resamples(modellist)
summary(results)
dotplot(results)

# Tuning mtry with best nodesize and ntree
node <- 3
modellist <- list()
set.seed(seed)
tunegrid <- expand.grid(.mtry=c(1:10))
for(m in c(1:10)){
  fit <- train(DBTS_NEW~., data = oriImp,
               method="rf", metric = "AUC", tuneGrid=tunegrid,
               nodesize = rbfnode, ntree = rbftree,
               trControl=control, .mtry = m)
  key <- toString(m)
  modellist[[key]] <- fit
}
results <- resamples(modellist)
summary(results)
dotplot(results)

fmtry <- 3

# Compare predicted results with original training set lables
set.seed(seed)
rfMdl2 <- randomForest(DBTS_NEW~., data = oriImp,
                      ntree = tree, nodesize = node, mtry = fmtry)
predrf2 <- predict(rfMdl2, oriImp)

# Confusion matrix of SVM + RF training model: rfcm
traincm2 <- confusionMatrix(predrf2, normTrain$DBTS_NEW)

svmrfpred2 <- predict(rfMdl2, normTest)
finalcm2 <- confusionMatrix(svmrfpred2, normTest$DBTS_NEW)

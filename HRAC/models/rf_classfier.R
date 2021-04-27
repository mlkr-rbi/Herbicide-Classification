# Load libraries
library(randomForest)
library(caret)

# Import data
h.train <- read.csv('../data/h_train.csv', row.names=1)

# Extending tunable hyperparameters
ext_RF <- list(type = "Classification", library = "randomForest", loop = NULL)
ext_RF$parameters <- data.frame(parameter = c("mtry", "ntree", "min_sample_split"), class = rep("numeric", 3), label = c("mtry", "ntree", "min_sample_split"))
ext_RF$grid <- function(x, y, len = NULL, search = "grid") {}
ext_RF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, mtry = param$mtry, ntree=param$ntree, min_sample_split=param$min_sample_split...)
}
ext_RF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
   predict(modelFit, newdata)
ext_RF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
   predict(modelFit, newdata, type = "prob")
ext_RF$sort <- function(x) x[order(x[,1]),]
ext_RF$levels <- function(x) x$classes

# Set trainControl
rf.ctrl <- trainControl(method="repeatedcv", number=10, repeats=10, returnResamp = "final", savePredictions = "final")

# Set up a tuning grid
rf.grid <- expand.grid(.mtry=c(seq(2,34,3)), 
                       .ntree=c(100, 500, 1000, 1500, 2000), 
                       .min_sample_split=c(2, 4, 8, 10))

# Hyperparameters tuning
set.seed(111)
hrac_rf <- train(factor(HRAC2020_class) ~ ., h.train, 
                 method = ext_RF,
                 tuneGrid = rf.grid,
                 metric = "Accuracy",
                 trControl = rf.ctrl)

plot(hrac_rf)

# Train model on whole train set with optimal hyperparameters
set.seed(111)
hrac_rf.fin <- train(factor(HRAC2020_class) ~ ., h.train, 
                     method = ext_RF,
                     tuneGrid = hrac_rf$bestTune,
                     metric = "Accuracy",
                     trControl = trainControl(method = "none"))

saveRDS(hrac_rf.fin, "./hrac_rf_model.rds")

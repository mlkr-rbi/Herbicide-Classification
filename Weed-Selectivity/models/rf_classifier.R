library(randomForest)
library(caret)

# EXTENDING CARET HYPERPARAMETER TUNING
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
set.seed(111)
rf.ctrl <- trainControl(method="repeatedcv", number=10, repeats=10, returnResamp = "final", savePredictions = "final")

# Set hyperparameter grid
rf.grid <- expand.grid(.mtry=c(2,4,6,8), 
                        .ntree=c(100, 500, 1000, 1500, 2000), 
                        .min_sample_split=c(4, 8, 10, 14))

# LogP dataset
# Hyperparameters tuning
set.seed(111)
selP_rf <- train(factor(Selectivity) ~., data = s.train.logP,
             method = ext_RF,
             tuneGrid = rf.grid,
             metric = "Accuracy",
             trControl = rf.ctrl)

plot(selP_rf)


# Train model on whole train set with optimal hyperparameters
set.seed(111)
selP_rf.fin <- train(factor(Selectivity) ~., data = s.train.logP,
                    method = ext_RF,
                    tuneGrid = selP_rf$bestTune,
                    metric = "Accuracy",
                    trControl = trainControl(method = "none"))
                    
# Save pretrained model
saveRDS(selP_rf.fin, "./wsel_logp_rf_model.rds")


# LogD dataset
# Hyperparameters tuning
set.seed(111)
selD_rf <- train(factor(Selectivity) ~., data = s.train.logD,
             method = ext_RF,
             tuneGrid = rf.grid,
             metric = "Accuracy",
             trControl = rf.ctrl)

plot(selD_rf)


# Train model on whole train set with optimal hyperparameters
set.seed(111)
selD_rf.fin <- train(factor(Selectivity) ~., data = s.train.logD,
                    method = ext_RF,
                    tuneGrid = selD_rf$bestTune,
                    metric = "Accuracy",
                    trControl = trainControl(method = "none"))
                    
# Save pretrained model
saveRDS(selD_rf.fin, "./wsel_logd_rf_model.rds")

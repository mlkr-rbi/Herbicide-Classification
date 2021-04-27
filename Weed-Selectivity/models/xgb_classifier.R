# Load libraries
library(randomForest)
library(caret)

# Import data
s.train.logD <- read.csv('../data/LogD/s_train_logD.csv', row.names=1)
s.train.logP <- read.csv('../data/LogP/s_train_logP.csv', row.names=1)

# Set trainControl
xgb.ctrl <- trainControl(method = "repeatedcv", repeats = 10, number = 10, returnResamp = "final", savePredictions = "final")

# Set hyperparameter grid
xgb.grid <- expand.grid(nrounds = c(100, 500, 1000, 1500), 
                        eta = c(0.01, 0.05, 0.1, 0.3), 
                        max_depth = c(4,6,10, 14),
                        gamma = 0,
                        colsample_bytree = 0.5,
                        min_child_weight = 1,
                        subsample = 1)


# LOG P Dataset
# Hyperparameters tuning
set.seed(111)
selP_xgb <- train(factor(Selectivity) ~., data = s.train.logP,
                 method="xgbTree",
                 trControl=xgb.ctrl,
                 tuneGrid=xgb.grid,
                 metric="Accuracy")

plot(selP_xgb)

# Train model on on whole train set with optimal hyperparameters
set.seed(111)
selP_xgb.fin <- train(factor(Selectivity) ~., data = s.train.logP,
                 method="xgbTree",
                 trControl=trainControl(method = "none"),
                 tuneGrid=selP_xgb$bestTune,
                 metric="Accuracy")

saveRDS(selP_xgb.fin, "./wsel_logp_xgb_model.rds")


# LOG D Dataset
# Hyperparameters tuning
set.seed(111)
selD_xgb <- train(factor(Selectivity) ~., data = s.train.logD,
                 method="xgbTree",
                 trControl=xgb.ctrl,
                 tuneGrid=xgb.grid,
                 metric="Accuracy")

plot(selD_xgb)

# Train model on on whole train set with optimal hyperparameters
set.seed(111)
selD_xgb.fin <- train(factor(Selectivity) ~., data = s.train.logD,
                 method="xgbTree",
                 trControl=trainControl(method = "none"),
                 tuneGrid=selD_xgb$bestTune,
                 metric="Accuracy")

saveRDS(selD_xgb.fin, "./wsel_logd_xgb_model.rds")

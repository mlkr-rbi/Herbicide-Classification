# Load libraries
library(caret)

# Import data
h.train <- read.csv('../data/h_train.csv', row.names=1)

# Set trainControl
xgb.ctrl <- trainControl(method = "repeatedcv", repeats = 10, number = 10,  returnResamp = "final", savePredictions = "final")

# Set up a tuning grid
xgb.grid <- expand.grid(nrounds = c(100, 500, 1000, 1500), 
                        eta = c(0.01, 0.05, 0.1, 0.3), 
                        max_depth = c(4,6,10, 14),
                        gamma = 0,
                        colsample_bytree = 0.5,
                        min_child_weight = 1,
                        subsample = 1)

# Hyperparameters tuning
set.seed(111)
hrac_xgb <- train(factor(HRAC2020_class) ~., data = h.train,
                  method="xgbTree",
                  trControl=xgb.ctrl,
                  tuneGrid=xgb.grid,
                  metric="Accuracy")

plot(hrac_xgb)


# Train model on on whole train set with optimal hyperparameters
set.seed(111)
hrac_xgb.fin <- train(factor(HRAC2020_class) ~., data = h.train,
                      method="xgbTree",
                      trControl=trainControl(method="none"),
                      tuneGrid=hrac_xgb$bestTune, 
                      metric="Accuracy")

saveRDS(hrac_xgb.fin, "./hrac_xgb_model.rds")

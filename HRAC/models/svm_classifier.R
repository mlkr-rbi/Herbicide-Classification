# Load libraries
library(caret)

# Import data
h.train <- read.csv('../data/h_train.csv', row.names=1)

# Set trainControl
svm.ctrl <- trainControl(method="repeatedcv", number=10, repeats=10, returnResamp = "final", savePredictions = "final")

# Set up a tuning grid
svm.grid <- expand.grid(sigma=2^c(-15,-10, -8, -6, -5), 
                        C=2^c(0:5))

# Hyperparameters tuning
set.seed(111)
hrac_svm <- train(factor(HRAC2020_class) ~., data = h.train, 
                  method = "svmRadial", 
                  trControl = svm.ctrl,  
                  metric = "Accuracy",
                  tuneGrid = svm.grid)

plot(hrac_svm)


# Train model on whole train set with optimat hyperparameters
set.seed(111)
hrac_svm.fin <- train(factor(HRAC2020_class) ~., data = h.train, 
                      method = "svmRadial", 
                      trControl = trainControl(method="none"),
                      metric = "Accuracy",
                      tuneGrid = hrac_svm$bestTune)

saveRDS(hrac_svm.fin, "./hrac_svm_model.rds")

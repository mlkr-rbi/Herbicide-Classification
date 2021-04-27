# Load libraries
library(randomForest)
library(caret)

# Import data
s.train.logD <- read.csv('../data/LogD/s_train_logD.csv', row.names=1)
s.train.logP <- read.csv('../data/LogP/s_train_logP.csv', row.names=1)

# Set trainControl
set.seed(111)
svm.ctrl <- trainControl(method="repeatedcv", number=10, repeats=10,  returnResamp = "final", savePredictions = "final")

# Set up a grid
svm.grid <- expand.grid(sigma= 2^c(-10, -5, -4, -2, -1), C= 2^c(0:5))

# LOG P
# Hyperparamter tuning
set.seed(111)
selP_svm <- train(factor(Selectivity) ~., data = s.train.logP,
              method = "svmRadial", 
              trControl = svm.ctrl,  
              metric = "Accuracy",
              tuneGrid = svm.grid)
plot(selP_svm)

# Train model on on whole train set with optimal hyperparameters
set.seed(111)
selP_svm.fin <- train(factor(Selectivity) ~., data = s.train.logP,
                     method = "svmRadial", 
                     trControl = trainControl(method="none"),
                     tuneGrid = selP_svm$bestTune,
                     metric = "Accuracy")
                     
saveRDS(selP_svm.fin, "./wsel_logp_svm_model.rds")


# LOG D
# Hyperparamter tuning
set.seed(111)
selD_svm <- train(factor(Selectivity) ~., data = s.train.logD,
              method = "svmRadial", 
              trControl = svm.ctrl,  
              #preProcess = c("center","scale"),
              metric = "Accuracy",
              tuneGrid = svm.grid)
plot(selD_svm)


# Train model on on whole train set with optimal hyperparameters
set.seed(111)
selD_svm.fin <- train(factor(Selectivity) ~., data = s.train.logD,
                     method = "svmRadial", 
                     trControl = trainControl(method="none"),
                     tuneGrid = selD_svm$bestTune,
                     metric = "Accuracy")
                     
                    
saveRDS(selD_svm.fin, "./wsel_logd_svm_model.rds")





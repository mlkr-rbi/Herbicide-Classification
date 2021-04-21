# Set trainControl
nb.ctrl <- trainControl(method="repeatedcv", number=10, repeats=10, returnResamp = "final", savePredictions = "final")

# Set hyperparameter grid
library(klaR)
grid <-  expand.grid(fL=c(0,0.3,0.5,0.7,1.0), usekernel = TRUE, adjust=c(0.1,0.3,0.5,0.7,1.0))

# LOG P
# Tune hyperparameters and evaluate model 
set.seed(111)
selP_nb <- train(factor(Selectivity) ~., data = s.train.logD,
             method = 'nb',
             tuneGrid = grid,
             trControl=nb.ctrl)
plot(selP_nb)

# Train model on on whole train set with optimal hyperparameters
set.seed(111)
selP_nb.fin <- train(factor(Selectivity) ~., data = s.train.logD,
             method = 'nb',
             tuneGrid = selP_nb$bestTune,
             trControl=trainControl(method = "none"))
             
saveRDS(selP_nb.fin, "./wsel_logp_nb_model.rds")
   
             
# LOG D
# Tune hyperparameters and evaluate model
set.seed(111)
selD_nb <- train(factor(Selectivity) ~., data = s.train.logD,
             method = 'nb',
             tuneGrid = grid,
             trControl=nb.ctrl)
plot(selD_nb)


# Train model on on whole train set with optimal hyperparameters
set.seed(111)
selD_nb.fin <- train(factor(Selectivity) ~., data = s.train.logD,
             method = 'nb',
             tuneGrid = selD_nb$bestTune,
             trControl=trainControl(method = "none"))

saveRDS(selD_nb.fin, "./wsel_logd_nb_model.rds")

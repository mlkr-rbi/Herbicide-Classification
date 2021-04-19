library(klaR)

# Set trainControl
nb.ctrl <- trainControl(method="repeatedcv", number=10, repeats=10, returnResamp = "final", savePredictions = "final") # , sampling = "up"

# Set up a tuning grid
grid <-  expand.grid(fL=c(0,0.5,1.0), 
                     usekernel = TRUE, 
                     adjust=c(0.1,0.5,1.0))


# Hyperparameters tuning
set.seed(111)
hrac_nb <- train(factor(HRAC2020_class) ~., data = h.train, 
                 method = 'nb',
                 tuneGrid = grid,
                 trControl=nb.ctrl,
                 metric = "Accuracy")

plot(hrac_nb)


# Train model on whole train set with optimal hyperparameters
set.seed(111)
hrac_nb.fin <- train(factor(HRAC2020_class) ~., data = h.train, 
                     method = 'nb',
                     tuneGrid = hrac_nb$bestTune,
                     trControl=trainControl(method="none"),
                     metric = "Accuracy")

saveRDS(hrac_nb.fin, "./hrac_nb_model.rds")
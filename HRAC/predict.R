# Import pretrained models (HRAC Classifiers)
hrac_rf.fin <- readRDS("models/hrac_rf.fin.rds")
hrac_xgb.fin <- readRDS("models/hrac_xgb.fin.rds")
hrac_svm.fin <- readRDS("models/hrac_svm.fin.rds")
hrac_nb.fin <- readRDS("models/hrac_nb.fin.rds")

# Import data
h.train <- read.csv('data/h_train.csv', row.names=1)
h.test <- read.csv('data/h_test.csv', row.names=1)
h.rest <- read.csv('data/h_rest.csv', row.names=1)
h.zclass <- read.csv('data/h_zclass.csv', row.names=1)
h.np <- read.csv('data/h_np.csv', row.names=1)

# List of datasets (HRAC) 
hrac_datasets<- list(h.train, h.test, h.rest, h.zclass, h.np) 
names(hrac_datasets) <- c("h.train", "h.test", "h.rest", "h.zclass", "h.np") 

# List of pretrained models
hrac_models <- list(hrac_rf.fin, hrac_xgb.fin, hrac_svm.fin, hrac_nb.fin)
names(hrac_models) <- c("hrac_rf.fin", "hrac_xgb.fin", "hrac_svm.fin", "hrac_nb.fin") 


# Empty list for HRAC predictions
hrac_predictions <- list(RF = list(train = list(), test = list(), rest = list(), z = list(), np = list()), 
                         XGB = list(train = list(), test = list(), rest = list(), z = list(), np = list()), 
                         SVM = list(train = list(), test = list(), rest = list(), z = list(), np = list()), 
                         NB = list(train = list(), test = list(), rest = list(), z = list(), np = list()))

# Predict labels and class probabilities for each dataset 
          
  for (i in 1:length(hrac_datasets)) {
      # RF
      hrac_predictions[["RF"]][[i]][[1]] <- predict(hrac_models[["hrac_rf.fin"]], hrac_datasets[[i]][,-1])
      names(hrac_predictions[["RF"]][[i]])[[1]] <- paste0(names(hrac_datasets)[[i]], sep="_", "pred")
  
      hrac_predictions[["RF"]][[i]][[2]] <- predict(hrac_models[["hrac_rf.fin"]], hrac_datasets[[i]][,-1], "prob")
      names(hrac_predictions[["RF"]][[i]])[[2]] <- paste0(names(hrac_datasets)[[i]], sep="_", "prob")
      
      # XGB
      hrac_predictions[["XGB"]][[i]][[1]] <- predict(hrac_models[["hrac_xgb.fin"]], hrac_datasets[[i]][,-1])
      names(hrac_predictions[["XGB"]][[i]])[[1]] <- paste0(names(hrac_datasets)[[i]], sep="_", "pred")
      
      hrac_predictions[["XGB"]][[i]][[2]] <- predict(hrac_models[["hrac_xgb.fin"]], hrac_datasets[[i]][,-1], "prob")
      names(hrac_predictions[["XGB"]][[i]])[[2]] <- paste0(names(hrac_datasets)[[i]], sep="_", "prob")
      
      # SVM
      hrac_predictions[["SVM"]][[i]][[1]] <- predict(hrac_models[["hrac_svm.fin"]], hrac_datasets[[i]][,-1])
      names(hrac_predictions[["SVM"]][[i]])[[1]] <- paste0(names(hrac_datasets)[[i]], sep="_", "pred")
      
      # NB
      hrac_predictions[["NB"]][[i]][[1]] <- predict(hrac_models[["hrac_nb.fin"]], hrac_datasets[[i]][,-1])
      names(hrac_predictions[["NB"]][[i]])[[1]] <- paste0(names(hrac_datasets)[[i]], sep="_", "pred")
      
      hrac_predictions[["NB"]][[i]][[2]] <- predict(hrac_models[["hrac_nb.fin"]], hrac_datasets[[i]][,-1], "prob")
      names(hrac_predictions[["NB"]][[i]])[[2]] <- paste0(names(hrac_datasets)[[i]], sep="_", "prob")
    
  }

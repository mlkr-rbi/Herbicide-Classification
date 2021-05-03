library(caret)

# Import pretrained models (HRAC Classifiers)
hrac_rf.fin <- readRDS("models/hrac_rf_model.rds")
hrac_xgb.fin <- readRDS("models/hrac_xgb_model.rds")
hrac_svm.fin <- readRDS("models/hrac_svm_model.rds")
hrac_nb.fin <- readRDS("models/hrac_nb_model.rds")

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

# A) Predict labels and class probabilities for each dataset 
          
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



# Write files - "labels"
write.csv(hrac_predictions[["RF"]][["train"]][["h.train_pred"]], "output/rf/h_train_pred.csv")
write.csv(hrac_predictions[["SVM"]][["train"]][["h.train_pred"]], "output/svm/h_train_pred.csv")
write.csv(hrac_predictions[["XGB"]][["train"]][["h.train_pred"]], "output/xgb/h_train_pred.csv")
write.csv(hrac_predictions[["NB"]][["train"]][["h.train_pred"]], "output/nb/h_train_pred.csv")


# Write files - "probs"
write.csv(hrac_predictions[["RF"]][["train"]][["h.train_prob"]], "output/rf/h_train_prob.csv")
write.csv(hrac_predictions[["SVM"]][["train"]][["h.train_prob"]], "output/svm/h_train_prob.csv")
write.csv(hrac_predictions[["XGB"]][["train"]][["h.train_prob"]], "output/xgb/h_train_prob.csv")
write.csv(hrac_predictions[["NB"]][["train"]][["h.train_prob"]], "output/nb/h_train_prob.csv")


# B) Confusion matrix
hrac.conf.ls <- list(RF=list(), XGB=list(), SVM=list(), NB=list())
           
for (i in 1:2) {
  hrac.conf.ls[["RF"]][[i]] <- confusionMatrix(hrac_predictions[["RF"]][[i]][[1]], factor(hrac_datasets[[i]]$HRAC2020_class))
  hrac.conf.ls[["XGB"]][[i]] <- confusionMatrix(hrac_predictions[["XGB"]][[i]][[1]], factor(hrac_datasets[[i]]$HRAC2020_class))
  hrac.conf.ls[["SVM"]][[i]] <- confusionMatrix(hrac_predictions[["SVM"]][[i]][[1]], factor(hrac_datasets[[i]]$HRAC2020_class))
  hrac.conf.ls[["NB"]][[i]] <- confusionMatrix(hrac_predictions[["NB"]][[i]][[1]], factor(hrac_datasets[[i]]$HRAC2020_class))
            
  names(hrac.conf.ls[["RF"]])[i] <- names(hrac_datasets)[i]
  names(hrac.conf.ls[["XGB"]])[i] <- names(hrac_datasets)[i]
  names(hrac.conf.ls[["SVM"]])[i] <- names(hrac_datasets)[i]
  names(hrac.conf.ls[["NB"]])[i] <- names(hrac_datasets)[i]

}

# Write files - "overall statistics"
write.csv(hrac.conf.ls[["RF"]][["h.train"]]$overall, "output/rf/h_train_stat.csv")
write.csv(hrac.conf.ls[["RF"]][["h.test"]]$overall, "output/rf/h_test_stat.csv")

write.csv(hrac.conf.ls[["XGB"]][["h.train"]]$overall, "output/xgb/h_train_stat.csv")
write.csv(hrac.conf.ls[["XGB"]][["h.test"]]$overall, "output/xgb/h_test_stat.csv")

write.csv(hrac.conf.ls[["SVM"]][["h.train"]]$overall, "output/svm/h_train_stat.csv")
write.csv(hrac.conf.ls[["SVM"]][["h.test"]]$overall, "output/svm/h_test_stat.csv")

write.csv(hrac.conf.ls[["NB"]][["h.train"]]$overall, "output/nb/h_train_stat.csv")
write.csv(hrac.conf.ls[["NB"]][["h.test"]]$overall, "output/nb/h_test_stat.csv")

# Write files - "spec by class"
write.csv(hrac.conf.ls[["RF"]][["h.train"]]$byClass , "output/rf/h_train_byClass.csv")
write.csv(hrac.conf.ls[["RF"]][["h.test"]]$byClass, "output/rf/h_test_byClass.csv")

write.csv(hrac.conf.ls[["XGB"]][["h.train"]]$byClass , "output/xgb/h_train_byClass.csv")
write.csv(hrac.conf.ls[["XGB"]][["h.test"]]$byClass, "output/xgb/h_test_byClass.csv")

write.csv(hrac.conf.ls[["SVM"]][["h.train"]]$byClass , "output/svm/h_train_byClass.csv")
write.csv(hrac.conf.ls[["SVM"]][["h.test"]]$byClass, "output/svm/h_test_byClass.csv")

write.csv(hrac.conf.ls[["NB"]][["h.train"]]$byClass , "output/nb/h_train_byClass.csv")
write.csv(hrac.conf.ls[["NB"]][["h.test"]]$byClass, "output/nb/h_test_byClass.csv")
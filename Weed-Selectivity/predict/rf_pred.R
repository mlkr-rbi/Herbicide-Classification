library(caret)

# Import data - LogD
s.train.logD <- read.csv('../data/LogD/s_train_logD.csv', row.names=1)
s.test.logD <- read.csv('../data/LogD/s_test_logD.csv', row.names=1)
s.cases.logD <- read.csv('../data/LogD/s_cases_logD.csv', row.names=1)
s.np.logD <- read.csv('../data/LogD/s_np_logD.csv', row.names=1)

# Import data - LogP
s.train.logP <- read.csv('../data/LogP/s_train_logP.csv', row.names=1)
s.test.logP <- read.csv('../data/LogP/s_test_logP.csv', row.names=1)
s.cases.logP <- read.csv('../data/LogP/s_cases_logP.csv', row.names=1)
s.np.logP <- read.csv('../data/LogP/s_np_logP.csv', row.names=1)

# List of test sets
sel_ts.ls <- list(list(s.train.logD, s.test.logD, s.cases.logD, s.np.logD), 
                  list(s.train.logP, s.test.logP, s.cases.logP, s.np.logP))

names(sel_ts.ls) <- c("LogD", "LogP") 
names(sel_ts.ls[["LogD"]]) <- c("s.train.logD", "s.test.logD", "s.cases.logD", "s.np.logD")
names(sel_ts.ls[["LogP"]]) <- c("s.train.logP", "s.test.logP", "s.cases.logP", "s.np.logP")

# Import pretrained model
selD_rf.fin <- readRDS("../models/wsel_logd_rf_model.rds")
selP_rf.fin <- readRDS("../models/wsel_logp_rf_model.rds")

# A) Predict for each dataset - "label"
sRF.pred.ls <- list(LogD=list(), LogP=list()) 

for (i in 1:4) {
   sRF.pred.ls[["LogD"]][[i]] <- predict(selD_rf.fin, sel_ts.ls[["LogD"]][[i]][,-1])
   sRF.pred.ls[["LogP"]][[i]] <- predict(selP_rf.fin, sel_ts.ls[["LogP"]][[i]][,-1])

    names(sRF.pred.ls[["LogD"]])[i] <- names(sel_ts.ls[["LogD"]])[i]
    names(sRF.pred.ls[["LogP"]])[i] <- names(sel_ts.ls[["LogP"]])[i]
}

# Write files - "labels"
write.csv(sRF.pred.ls[["LogD"]][[1]], 'output/rf/logD/rf_s_train_pred.csv')
write.csv(sRF.pred.ls[["LogD"]][[2]], 'output/rf/logD/rf_s_test_pred.csv')
write.csv(sRF.pred.ls[["LogD"]][[3]], 'output/rf/logD/rf_s_cases_pred.csv')
write.csv(sRF.pred.ls[["LogD"]][[4]], 'output/rf/logD/rf_s_np_pred.csv')

write.csv(sRF.pred.ls[["LogP"]][[1]], 'output/rf/logP/rf_s_train_pred.csv')
write.csv(sRF.pred.ls[["LogP"]][[2]], 'output/rf/logP/rf_s_test_pred.csv')
write.csv(sRF.pred.ls[["LogP"]][[3]], 'output/rf/logP/rf_s_cases_pred.csv')
write.csv(sRF.pred.ls[["LogP"]][[4]], 'output/rf/logP/rf_s_np_pred.csv')

          
          
# B) Predict for each dataset - "prob"
sRF.pred_p.ls <- list(LogD=list(), LogP=list())
          
for (i in 1:4) {
    sRF.pred_p.ls[["LogD"]][[i]] <- predict(selD_rf.fin, sel_ts.ls[["LogD"]][[i]][,-1], "prob")
    sRF.pred_p.ls[["LogP"]][[i]] <- predict(selP_rf.fin, sel_ts.ls[["LogP"]][[i]][,-1], "prob")

    names(sRF.pred_p.ls[["LogD"]])[i] <- names(sel_ts.ls[["LogD"]])[i]
    names(sRF.pred_p.ls[["LogP"]])[i] <- names(sel_ts.ls[["LogP"]])[i]

}


# Write files - "probs"
write.csv(sRF.pred_p.ls[["LogD"]][[1]], 'output/rf/logD/rf_s_train_pred_p.csv')
write.csv(sRF.pred_p.ls[["LogD"]][[2]], 'output/rf/logD/rf_s_test_pred_p.csv')
write.csv(sRF.pred_p.ls[["LogD"]][[3]], 'output/rf/logD/rf_s_cases_pred_p.csv')
write.csv(sRF.pred_p.ls[["LogD"]][[4]], 'output/rf/logD/rf_s_np_pred_p.csv')

write.csv(sRF.pred_p.ls[["LogP"]][[1]], 'output/rf/logP/rf_s_train_pred_p.csv')
write.csv(sRF.pred_p.ls[["LogP"]][[2]], 'output/rf/logP/rf_s_test_pred_p.csv')
write.csv(sRF.pred_p.ls[["LogP"]][[3]], 'output/rf/logP/rf_s_cases_pred_p.csv')
write.csv(sRF.pred_p.ls[["LogP"]][[4]], 'output/rf/logP/rf_s_np_pred_p.csv')


# C) Confusion matrix
sRF.conf.ls <- list(LogD=list(), LogP=list())
          
for (i in 1:2) {
    sRF.conf.ls[["LogD"]][[i]] <- confusionMatrix(sRF.pred.ls[["LogD"]][[i]], factor(sel_ts.ls[["LogD"]][[i]]$Selectivity))
    sRF.conf.ls[["LogP"]][[i]] <- confusionMatrix(sRF.pred.ls[["LogP"]][[i]], factor(sel_ts.ls[["LogP"]][[i]]$Selectivity))
            
    names(sRF.conf.ls[["LogD"]])[i] <- names(sel_ts.ls[["LogD"]])[i]
    names(sRF.conf.ls[["LogP"]])[i] <- names(sel_ts.ls[["LogP"]])[i]

}


# Write files - "overall statistics"
write.csv(sRF.conf.ls[["LogD"]][[1]]$overall, 'output/rf/logD/s_train_stat.csv')
write.csv(sRF.conf.ls[["LogD"]][[2]]$overall, 'output/rf/logD/s_test_stat.csv')

write.csv(sRF.conf.ls[["LogP"]][[1]]$overall, 'output/rf/logP/s_train_stat.csv')
write.csv(sRF.conf.ls[["LogP"]][[2]]$overall, 'output/rf/logP/s_test_stat.csv')

# Write files - "spec by class"
write.csv(sRF.conf.ls[["LogD"]][[1]]$byClass, 'output/rf/logD/s_train_byClass.csv')
write.csv(sRF.conf.ls[["LogD"]][[2]]$byClass, 'output/rf/logD/s_test_byClass.csv')

write.csv(sRF.conf.ls[["LogP"]][[1]]$byClass, 'output/rf/logP/s_train_byClass.csv')
write.csv(sRF.conf.ls[["LogP"]][[2]]$byClass, 'output/rf/logP/s_test_byClass.csv')

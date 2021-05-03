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
selD_xgb.fin <- readRDS("../models/wsel_logd_xgb_model.rds")
selP_xgb.fin <- readRDS("../models/wsel_logp_xgb_model.rds")


# A) Predict for each dataset - "label"
sXGB.pred.ls <- list(LogD=list(), LogP=list()) 

for (i in 1:4) {
    sXGB.pred.ls[["LogD"]][[i]] <- predict(selD_xgb.fin, sel_ts.ls[["LogD"]][[i]][,-1])
    sXGB.pred.ls[["LogP"]][[i]] <- predict(selP_xgb.fin, sel_ts.ls[["LogP"]][[i]][,-1])

    names(sXGB.pred.ls[["LogD"]])[i] <- names(sel_ts.ls[["LogD"]])[i]
    names(sXGB.pred.ls[["LogP"]])[i] <- names(sel_ts.ls[["LogP"]])[i]
}


# Write files - "labels"
write.csv(sXGB.pred.ls[["LogD"]][[1]], 'output/xgb/logD/xgb_s_train_pred.csv')
write.csv(sXGB.pred.ls[["LogD"]][[2]], 'output/xgb/logD/xgb_s_test_pred.csv')
write.csv(sXGB.pred.ls[["LogD"]][[3]], 'output/xgb/logD/xgb_s_cases_pred.csv')
write.csv(sXGB.pred.ls[["LogD"]][[4]], 'output/xgb/logD/xgb_s_np_pred.csv')

write.csv(sXGB.pred.ls[["LogP"]][[1]], 'output/xgb/logP/xgb_s_train_pred.csv')
write.csv(sXGB.pred.ls[["LogP"]][[2]], 'output/xgb/logP/xgb_s_test_pred.csv')
write.csv(sXGB.pred.ls[["LogP"]][[3]], 'output/xgb/logP/xgb_s_cases_pred.csv')
write.csv(sXGB.pred.ls[["LogP"]][[4]], 'output/xgb/logP/xgb_s_np_pred.csv')
          
# B) Predict for each dataset - "prob"
sXGB.pred_p.ls <- list(LogD=list(), LogP=list())
          
for (i in 1:4) {
    sXGB.pred_p.ls[["LogD"]][[i]] <- predict(selD_xgb.fin, sel_ts.ls[["LogD"]][[i]][,-1], "prob")
    sXGB.pred_p.ls[["LogP"]][[i]] <- predict(selP_xgb.fin, sel_ts.ls[["LogP"]][[i]][,-1], "prob")

    names(sXGB.pred_p.ls[["LogD"]])[i] <- names(sel_ts.ls[["LogD"]])[i]
    names(sXGB.pred_p.ls[["LogP"]])[i] <- names(sel_ts.ls[["LogP"]])[i]

}


# Write files - "probs"
write.csv(sXGB.pred_p.ls[["LogD"]][[1]], 'output/xgb/logD/xgb_s_train_pred_p.csv')
write.csv(sXGB.pred_p.ls[["LogD"]][[2]], 'output/xgb/logD/xgb_s_test_pred_p.csv')
write.csv(sXGB.pred_p.ls[["LogD"]][[3]], 'output/xgb/logD/xgb_s_cases_pred_p.csv')
write.csv(sXGB.pred_p.ls[["LogD"]][[4]], 'output/xgb/logD/xgb_s_np_pred_p.csv')

write.csv(sXGB.pred_p.ls[["LogP"]][[1]], 'output/xgb/logP/xgb_s_train_pred_p.csv')
write.csv(sXGB.pred_p.ls[["LogP"]][[2]], 'output/xgb/logP/xgb_s_test_pred_p.csv')
write.csv(sXGB.pred_p.ls[["LogP"]][[3]], 'output/xgb/logP/xgb_s_cases_pred_p.csv')
write.csv(sXGB.pred_p.ls[["LogP"]][[4]], 'output/xgb/logP/xgb_s_np_pred_p.csv')

          
# C) Confusion matrix
sXGB.conf.ls <- list(LogD=list(), LogP=list())
          
for (i in 1:2) {
    sXGB.conf.ls[["LogD"]][[i]] <- confusionMatrix(sXGB.pred.ls[["LogD"]][[i]], factor(sel_ts.ls[["LogD"]][[i]]$Selectivity))
    sXGB.conf.ls[["LogP"]][[i]] <- confusionMatrix(sXGB.pred.ls[["LogP"]][[i]], factor(sel_ts.ls[["LogP"]][[i]]$Selectivity))
            
    names(sXGB.conf.ls[["LogD"]])[i] <- names(sel_ts.ls[["LogD"]])[i]
    names(sXGB.conf.ls[["LogP"]])[i] <- names(sel_ts.ls[["LogP"]])[i]
}


# Write files - "overall statistics"
write.csv(sXGB.conf.ls[["LogD"]][[1]]$overall, 'output/xgb/logD/s_train_stat.csv')
write.csv(sXGB.conf.ls[["LogD"]][[2]]$overall, 'output/xgb/logD/s_test_stat.csv')

write.csv(sXGB.conf.ls[["LogP"]][[1]]$overall, 'output/xgb/logP/s_train_stat.csv')
write.csv(sXGB.conf.ls[["LogP"]][[2]]$overall, 'output/xgb/logP/s_test_stat.csv')

# Write files - "spec by class"
write.csv(sXGB.conf.ls[["LogD"]][[1]]$byClass, 'output/xgb/logD/s_train_byClass.csv')
write.csv(sXGB.conf.ls[["LogD"]][[2]]$byClass, 'output/xgb/logD/s_test_byClass.csv')

write.csv(sXGB.conf.ls[["LogP"]][[1]]$byClass, 'output/xgb/logP/s_train_byClass.csv')
write.csv(sXGB.conf.ls[["LogP"]][[2]]$byClass, 'output/xgb/logP/s_test_byClass.csv')

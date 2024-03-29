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
selD_svm.fin <- readRDS("../models/wsel_logd_svm_model.rds")
selP_svm.fin <- readRDS("../models/wsel_logp_svm_model.rds")


# A) Predict for each dataset - "label"
sSVM.pred.ls <- list(LogD=list(), LogP=list()) 

for (i in 1:4) {
   sSVM.pred.ls[["LogD"]][[i]] <- predict(selD_svm.fin, sel_ts.ls[["LogD"]][[i]][,-1])
   sSVM.pred.ls[["LogP"]][[i]] <- predict(selP_svm.fin, sel_ts.ls[["LogP"]][[i]][,-1])

    names(sSVM.pred.ls[["LogD"]])[i] <- names(sel_ts.ls[["LogD"]])[i]
    names(sSVM.pred.ls[["LogP"]])[i] <- names(sel_ts.ls[["LogP"]])[i]
}


# Write files - "labels"
write.csv(sSVM.pred.ls[["LogD"]][[1]], 'output/svm/logD/svm_s_train_pred.csv')
write.csv(sSVM.pred.ls[["LogD"]][[2]], 'output/svm/logD/svm_s_test_pred.csv')
write.csv(sSVM.pred.ls[["LogD"]][[3]], 'output/svm/logD/svm_s_cases_pred.csv')
write.csv(sSVM.pred.ls[["LogD"]][[4]], 'output/svm/logD/svm_s_np_pred.csv')

write.csv(sSVM.pred.ls[["LogP"]][[1]], 'output/svm/logP/svm_s_train_pred.csv')
write.csv(sSVM.pred.ls[["LogP"]][[2]], 'output/svm/logP/svm_s_test_pred.csv')
write.csv(sSVM.pred.ls[["LogP"]][[3]], 'output/svm/logP/svm_s_cases_pred.csv')
write.csv(sSVM.pred.ls[["LogP"]][[4]], 'output/svm/logP/svm_s_np_pred.csv')

     
          
# B) Confusion matrix
sSVM.conf.ls <- list(LogD=list(), LogP=list())
          
for (i in 1:2) {
    sSVM.conf.ls[["LogD"]][[i]] <- confusionMatrix(sSVM.pred.ls[["LogD"]][[i]], factor(sel_ts.ls[["LogD"]][[i]]$Selectivity))
    sSVM.conf.ls[["LogP"]][[i]] <- confusionMatrix(sSVM.pred.ls[["LogP"]][[i]], factor(sel_ts.ls[["LogP"]][[i]]$Selectivity))
            
    names(sSVM.conf.ls[["LogD"]])[i] <- names(sel_ts.ls[["LogD"]])[i]
    names(sSVM.conf.ls[["LogP"]])[i] <- names(sel_ts.ls[["LogP"]])[i]

}


# Write files - "overall statistics"
write.csv(sSVM.conf.ls[["LogD"]][[1]]$overall, 'output/svm/logD/s_train_stat.csv')
write.csv(sSVM.conf.ls[["LogD"]][[2]]$overall, 'output/svm/logD/s_test_stat.csv')

write.csv(sSVM.conf.ls[["LogP"]][[1]]$overall, 'output/svm/logP/s_train_stat.csv')
write.csv(sSVM.conf.ls[["LogP"]][[2]]$overall, 'output/svm/logP/s_test_stat.csv')

# Write files - "spec by class"
write.csv(sSVM.conf.ls[["LogD"]][[1]]$byClass, 'output/svm/logD/s_train_byClass.csv')
write.csv(sSVM.conf.ls[["LogD"]][[2]]$byClass, 'output/svm/logD/s_test_byClass.csv')

write.csv(sSVM.conf.ls[["LogP"]][[1]]$byClass, 'output/svm/logP/s_train_byClass.csv')
write.csv(sSVM.conf.ls[["LogP"]][[2]]$byClass, 'output/svm/logP/s_test_byClass.csv')

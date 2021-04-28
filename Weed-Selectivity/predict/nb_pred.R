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
selD_nb.fin <- readRDS("../models/wsel_logd_nb_model.rds")
selP_nb.fin <- readRDS("../models/wsel_logp_nb_model.rds")


# A) Predict for each dataset - "label"
sNB.pred.ls <- list(LogD=list(), LogP=list())   

for (i in 1:4) {
    sNB.pred.ls[["LogD"]][[i]] <- predict(selD_nb.fin, sel_ts.ls[["LogD"]][[i]][,-1])
    sNB.pred.ls[["LogP"]][[i]] <- predict(selP_nb.fin, sel_ts.ls[["LogP"]][[i]][,-1])

    names(sNB.pred.ls[["LogD"]])[i] <- names(sel_ts.ls[["LogD"]])[i]
    names(sNB.pred.ls[["LogP"]])[i] <- names(sel_ts.ls[["LogP"]])[i]

}


# Write files - "labels"
write.csv(sNB.pred.ls[["LogD"]][[1]], 'output/nb/logD/nb_s_train_pred.csv')
write.csv(sNB.pred.ls[["LogD"]][[2]], 'output/nb/logD/nb_s_test_pred.csv')
write.csv(sNB.pred.ls[["LogD"]][[3]], 'output/nb/logD/nb_s_cases_pred.csv')
write.csv(sNB.pred.ls[["LogD"]][[4]], 'output/nb/logD/nb_s_np_pred.csv')

write.csv(sNB.pred.ls[["LogP"]][[1]], 'output/nb/logP/nb_s_train_pred.csv')
write.csv(sNB.pred.ls[["LogP"]][[2]], 'output/nb/logP/nb_s_test_pred.csv')
write.csv(sNB.pred.ls[["LogP"]][[3]], 'output/nb/logP/nb_s_cases_pred.csv')
write.csv(sNB.pred.ls[["LogP"]][[4]], 'output/nb/logP/nb_s_np_pred.csv')

          
# B) Predict for each dataset - "prob"
sNB.pred_p.ls <- list(LogD=list(), LogP=list())
          
for (i in 1:4) {
    sNB.pred_p.ls[["LogD"]][[i]] <- predict(selD_nb.fin, sel_ts.ls[["LogD"]][[i]][,-1], "prob")
    sNB.pred_p.ls[["LogP"]][[i]] <- predict(selP_nb.fin, sel_ts.ls[["LogP"]][[i]][,-1], "prob")

    names(sNB.pred_p.ls[["LogD"]])[i] <- names(sel_ts.ls[["LogD"]])[i]
    names(sNB.pred_p.ls[["LogP"]])[i] <- names(sel_ts.ls[["LogP"]])[i]

}


# Write files - "probs"
write.csv(sNB.pred_p.ls[["LogD"]][[1]], 'output/nb/logD/nb_s_train_pred_p.csv')
write.csv(sNB.pred_p.ls[["LogD"]][[2]], 'output/nb/logD/nb_s_test_pred_p.csv')
write.csv(sNB.pred_p.ls[["LogD"]][[3]], 'output/nb/logD/nb_s_cases_pred_p.csv')
write.csv(sNB.pred_p.ls[["LogD"]][[4]], 'output/nb/logD/nb_s_np_pred_p.csv')

write.csv(sNB.pred_p.ls[["LogP"]][[1]], 'output/nb/logP/nb_s_train_pred_p.csv')
write.csv(sNB.pred_p.ls[["LogP"]][[2]], 'output/nb/logP/nb_s_test_pred_p.csv')
write.csv(sNB.pred_p.ls[["LogP"]][[3]], 'output/nb/logP/nb_s_cases_pred_p.csv')
write.csv(sNB.pred_p.ls[["LogP"]][[4]], 'output/nb/logP/nb_s_np_pred_p.csv')

          
# C) Confusion matrix
sNB.conf.ls <- list(LogD=list(), LogP=list())
          
for (i in 1:2) {
    sNB.conf.ls[["LogD"]][[i]] <- confusionMatrix(sNB.pred.ls[["LogD"]][[i]], factor(sel_ts.ls[["LogD"]][[i]]$Selectivity))
    sNB.conf.ls[["LogP"]][[i]] <- confusionMatrix(sNB.pred.ls[["LogP"]][[i]], factor(sel_ts.ls[["LogP"]][[i]]$Selectivity))
            
    names(sNB.conf.ls[["LogD"]])[i] <- names(sel_ts.ls[["LogD"]])[i]
    names(sNB.conf.ls[["LogP"]])[i] <- names(sel_ts.ls[["LogP"]])[i]
}


# Write files - "overall statistics"
write.csv(sNB.conf.ls[["LogD"]][[1]]$overall, 'output/nb/logD/overall_statistics.csv')
write.csv(sNB.conf.ls[["LogD"]][[2]]$overall, 'output/nb/logD/overall_statistics.csv')

write.csv(sNB.conf.ls[["LogP"]][[1]]$overall, 'output/nb/logD/overall_statistics.csv')
write.csv(sNB.conf.ls[["LogP"]][[2]]$overall, 'output/nb/logD/overall_statistics.csv')

# Write files - "spec by class"
write.csv(sNB.conf.ls[["LogD"]][[1]]$byClass, 'output/nb/logP/spec_byClass.csv')
write.csv(sNB.conf.ls[["LogD"]][[2]]$byClass, 'output/nb/logP/spec_byClass.csv')

write.csv(sNB.conf.ls[["LogP"]][[1]]$byClass, 'output/nb/logP/spec_byClass.csv')
write.csv(sNB.conf.ls[["LogP"]][[2]]$byClass, 'output/nb/logP/spec_byClass.csv')


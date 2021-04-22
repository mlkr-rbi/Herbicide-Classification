# List of test sets
sel_ts.ls <- list(list(s.train.logD, s.test.logD, s.cases.logD, s.np.logD), 
                  list(s.train.logP, s.test.logP, s.cases.logP, s.np.logP))

names(sel_ts.ls) <- c("LogD", "LogP") 
names(sel_ts.ls[["LogD"]]) <- c("s.train.logD", "s.test.logD", "s.cases.logD", "s.np.logD")
names(sel_ts.ls[["LogP"]]) <- c("s.train.logP", "s.test.logP", "s.cases.logP", "s.np.logP")


# Import pretrained model
selD_rf.fin <- readRDS("./selD_rf.fin")
selP_rf.fin <- readRDS("./selP_rf.fin")


# A) Predict for each dataset - "label"
sRF.pred.ls <- list(LogD=list(), LogP=list()) 

for (i in 1:4) {
   sRF.pred.ls[["LogD"]][[i]] <- predict(selD_rf.fin, sel_ts.ls[["LogD"]][[i]][,-1])
   sRF.pred.ls[["LogP"]][[i]] <- predict(selP_rf.fin, sel_ts.ls[["LogP"]][[i]][,-1])

    names(sRF.pred.ls[["LogD"]])[i] <- names(sel_ts.ls[["LogD"]])[i]
    names(sRF.pred.ls[["LogP"]])[i] <- names(sel_ts.ls[["LogP"]])[i]
}
          
          
# B) Predict for each dataset - "prob"
sRF.pred_p.ls <- list(LogD=list(), LogP=list())
          
for (i in 1:4) {
    sRF.pred_p.ls[["LogD"]][[i]] <- predict(selD_rf.fin, sel_ts.ls[["LogD"]][[i]][,-1], "prob")
    sRF.pred_p.ls[["LogP"]][[i]] <- predict(selP_rf.fin, sel_ts.ls[["LogP"]][[i]][,-1], "prob")

    names(sRF.pred_p.ls[["LogD"]])[i] <- names(sel_ts.ls[["LogD"]])[i]
    names(sRF.pred_p.ls[["LogP"]])[i] <- names(sel_ts.ls[["LogP"]])[i]

}
          
# C) Confusion matrix
sRF.conf.ls <- list(LogD=list(), LogP=list())
          
for (i in 1:2) {
    sRF.conf.ls[["LogD"]][[i]] <- confusionMatrix(sRF.pred.ls[["LogD"]][[i]], factor(sel_ts.ls[["LogD"]][[i]]$Selectivity))
    sRF.conf.ls[["LogP"]][[i]] <- confusionMatrix(sRF.pred.ls[["LogP"]][[i]], factor(sel_ts.ls[["LogP"]][[i]]$Selectivity))
            
    names(sRF.conf.ls[["LogD"]])[i] <- names(sel_ts.ls[["LogD"]])[i]
    names(sRF.conf.ls[["LogP"]])[i] <- names(sel_ts.ls[["LogP"]])[i]

}

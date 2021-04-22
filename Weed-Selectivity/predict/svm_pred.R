# List of test sets
sel_ts.ls <- list(list(s.train.logD, s.test.logD, s.cases.logD, s.np.logD), 
                  list(s.train.logP, s.test.logP, s.cases.logP, s.np.logP))

names(sel_ts.ls) <- c("LogD", "LogP") 
names(sel_ts.ls[["LogD"]]) <- c("s.train.logD", "s.test.logD", "s.cases.logD", "s.np.logD")
names(sel_ts.ls[["LogP"]]) <- c("s.train.logP", "s.test.logP", "s.cases.logP", "s.np.logP")


# Import pretrained model
selD_svm.fin <- readRDS("./selD_svm.fin")
selP_svm.fin <- readRDS("./selP_svm.fin")


# A) Predict for each dataset - "label"
sSVM.pred.ls <- list(LogD=list(), LogP=list()) 

for (i in 1:4) {
   sSVM.pred.ls[["LogD"]][[i]] <- predict(selD_svm.fin, sel_ts.ls[["LogD"]][[i]][,-1])
   sSVM.pred.ls[["LogP"]][[i]] <- predict(selP_svm.fin, sel_ts.ls[["LogP"]][[i]][,-1])

    names(sSVM.pred.ls[["LogD"]])[i] <- names(sel_ts.ls[["LogD"]])[i]
    names(sSVM.pred.ls[["LogP"]])[i] <- names(sel_ts.ls[["LogP"]])[i]
}
     
          
# B) Confusion matrix
sSVM.conf.ls <- list(LogD=list(), LogP=list())
          
for (i in 1:2) {
    sSVM.conf.ls[["LogD"]][[i]] <- confusionMatrix(sSVM.pred.ls[["LogD"]][[i]], factor(sel_ts.ls[["LogD"]][[i]]$Selectivity))
    sSVM.conf.ls[["LogP"]][[i]] <- confusionMatrix(sSVM.pred.ls[["LogP"]][[i]], factor(sel_ts.ls[["LogP"]][[i]]$Selectivity))
            
    names(sSVM.conf.ls[["LogD"]])[i] <- names(sel_ts.ls[["LogD"]])[i]
    names(sSVM.conf.ls[["LogP"]])[i] <- names(sel_ts.ls[["LogP"]])[i]

}

# Import data - LogD
s.train.logD <- read.csv('../data/LogD/s_train_logD.csv', row.names=1)
s.test.logD <- read.csv('../data/LogD/s_test_logD.csv', row.names=1)
s.cases.logD <- read.csv('../data/LogD/s_cases_logD.csv', row.names=1)
s.np.logD <- read.csv('../data/LogD/s_np_logD.csv', row.names=1)

# Import data - LogP
s.train.logP <- read.csv('../data/LogD/s_train_logP.csv', row.names=1)
s.test.logP <- read.csv('../data/LogD/s_test_logP.csv', row.names=1)
s.cases.logP <- read.csv('../data/LogD/s_cases_logP.csv', row.names=1)
s.np.logP <- read.csv('../data/LogD/s_np_logP.csv', row.names=1)

# List of test sets
sel_ts.ls <- list(list(s.train.logD, s.test.logD, s.cases.logD, s.np.logD), 
                  list(s.train.logP, s.test.logP, s.cases.logP, s.np.logP))

names(sel_ts.ls) <- c("LogD", "LogP") 
names(sel_ts.ls[["LogD"]]) <- c("s.train.logD", "s.test.logD", "s.cases.logD", "s.np.logD")
names(sel_ts.ls[["LogP"]]) <- c("s.train.logP", "s.test.logP", "s.cases.logP", "s.np.logP")


# Import pretrained model
selD_xgb.fin <- readRDS("./selD_xgb.fin")
selP_xgb.fin <- readRDS("./selP_xgb.fin")


# A) Predict for each dataset - "label"
sXGB.pred.ls <- list(LogD=list(), LogP=list()) 

for (i in 1:4) {
    sXGB.pred.ls[["LogD"]][[i]] <- predict(selD_xgb.fin, sel_ts.ls[["LogD"]][[i]][,-1])
    sXGB.pred.ls[["LogP"]][[i]] <- predict(selP_xgb.fin, sel_ts.ls[["LogP"]][[i]][,-1])

    names(sXGB.pred.ls[["LogD"]])[i] <- names(sel_ts.ls[["LogD"]])[i]
    names(sXGB.pred.ls[["LogP"]])[i] <- names(sel_ts.ls[["LogP"]])[i]
}
          
# B) Predict for each dataset - "prob"
sXGB.pred_p.ls <- list(LogD=list(), LogP=list())
          
for (i in 1:4) {
    sXGB.pred_p.ls[["LogD"]][[i]] <- predict(selD_xgb.fin, sel_ts.ls[["LogD"]][[i]][,-1], "prob")
    sXGB.pred_p.ls[["LogP"]][[i]] <- predict(selP_xgb.fin, sel_ts.ls[["LogP"]][[i]][,-1], "prob")

    names(sXGB.pred_p.ls[["LogD"]])[i] <- names(sel_ts.ls[["LogD"]])[i]
    names(sXGB.pred_p.ls[["LogP"]])[i] <- names(sel_ts.ls[["LogP"]])[i]

}
          
# C) Confusion matrix
sXGB.conf.ls <- list(LogD=list(), LogP=list())
          
for (i in 1:2) {
    sXGB.conf.ls[["LogD"]][[i]] <- confusionMatrix(sXGB.pred.ls[["LogD"]][[i]], factor(sel_ts.ls[["LogD"]][[i]]$Selectivity))
    sXGB.conf.ls[["LogP"]][[i]] <- confusionMatrix(sXGB.pred.ls[["LogP"]][[i]], factor(sel_ts.ls[["LogP"]][[i]]$Selectivity))
            
    names(sXGB.conf.ls[["LogD"]])[i] <- names(sel_ts.ls[["LogD"]])[i]
    names(sXGB.conf.ls[["LogP"]])[i] <- names(sel_ts.ls[["LogP"]])[i]
}

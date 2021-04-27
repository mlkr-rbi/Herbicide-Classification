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
selD_nb.fin <- readRDS("./selD_nb.fin")
selP_nb.fin <- readRDS("./selP_nb.fin")


# A) Predict for each dataset - "label"
sNB.pred.ls <- list(LogD=list(), LogP=list())   

for (i in 1:3) {
    sNB.pred.ls[["LogD"]][[i]] <- predict(selD_nb.fin, sel_ts.ls[["LogD"]][[i]][,-1])
    sNB.pred.ls[["LogP"]][[i]] <- predict(selP_nb.fin, sel_ts.ls[["LogP"]][[i]][,-1])

    names(sNB.pred.ls[["LogD"]])[i] <- names(sel_ts.ls[["LogD"]])[i]
    names(sNB.pred.ls[["LogP"]])[i] <- names(sel_ts.ls[["LogP"]])[i]

}
          
# B) Predict for each dataset - "prob"
sNB.pred_p.ls <- list(LogD=list(), LogP=list())
          
for (i in 1:3) {
    sNB.pred_p.ls[["LogD"]][[i]] <- predict(selD_nb.fin, sel_ts.ls[["LogD"]][[i]][,-1], "prob")
    sNB.pred_p.ls[["LogP"]][[i]] <- predict(selP_nb.fin, sel_ts.ls[["LogP"]][[i]][,-1], "prob")

    names(sNB.pred_p.ls[["LogD"]])[i] <- names(sel_ts.ls[["LogD"]])[i]
    names(sNB.pred_p.ls[["LogP"]])[i] <- names(sel_ts.ls[["LogP"]])[i]

}
          
# C) Confusion matrix
sNB.conf.ls <- list(LogD=list(), LogP=list())
          
for (i in 1:2) {
    sNB.conf.ls[["LogD"]][[i]] <- confusionMatrix(sNB.pred.ls[["LogD"]][[i]], factor(sel_ts.ls[["LogD"]][[i]]$Selectivity))
    sNB.conf.ls[["LogP"]][[i]] <- confusionMatrix(sNB.pred.ls[["LogP"]][[i]], factor(sel_ts.ls[["LogP"]][[i]]$Selectivity))
            
    names(sNB.conf.ls[["LogD"]])[i] <- names(sel_ts.ls[["LogD"]])[i]
    names(sNB.conf.ls[["LogP"]])[i] <- names(sel_ts.ls[["LogP"]])[i]
}

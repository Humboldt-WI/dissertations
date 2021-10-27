# PREPROCESSED DATA PROCESSING

#setwd("C:/Users/Johannes/OneDrive/Dokumente/Humboldt-Universität/Msc WI/1_4. Sem/Master Thesis II/")
rm(list = ls());gc()
set.seed(0)
options(scipen=999)

# - load packages

packages <- c("pROC", "EMP")
sapply(packages, require, character.only = TRUE)
rm(packages)

# - read data set (see Fairness Definitions Explained)
dtest_unscaled <- read.csv("taiwan_orig_test.csv")
dtest_unscaled <- subset(dtest_unscaled, select = c(CREDIT_AMNT,AGE, TARGET))

# check fairness in training set
source("95_fairnessMetrics.R")

# results for base scenario
# Add base scenario = all get loan
AUC <- as.numeric(roc(dtest_unscaled$TARGET, rep.int(1, nrow(dtest_unscaled)))$auc)
EMP <- NA
acceptedLoans <- 1
loanprofit <- NULL
for (i in 1:nrow(dtest_unscaled)){
  p = ifelse(dtest_unscaled$TARGET[i]=="Bad", -dtest_unscaled$CREDIT_AMNT[i], dtest_unscaled$CREDIT_AMNT[i] * 0.2644)
  loanprofit <- c(loanprofit, p)
}
profit <- sum(loanprofit)
profitPerLoan <- profit/nrow(dtest_unscaled)

# fairness criteria average
statParityDiff <- statParDiff(sens.attr = dtest_unscaled$AGE, target.attr = rep("Good", nrow(dtest_unscaled)))
averageOddsDiff <- avgOddsDiff(sens.attr = dtest_unscaled$AGE, target.attr = dtest_unscaled$TARGET, predicted.attr = rep("Good", nrow(dtest_unscaled)))
predParityDiff <- predParDiff(sens.attr = dtest_unscaled$AGE, target.attr = dtest_unscaled$TARGET, predicted.attr = rep("Good", nrow(dtest_unscaled)))

cm <- confusionMatrix(data = as.factor(rep("Good", nrow(dtest_unscaled))), reference = dtest_unscaled$TARGET)
balAccuracy <- cm$byClass[['Balanced Accuracy']]

test_eval <- rbind(AUC, balAccuracy, EMP, acceptedLoans, profit, profitPerLoan, statParityDiff, averageOddsDiff, predParityDiff)
test_eval

write.csv(test_eval, "BASE_Results.csv", row.names = T)


rm(acceptedLoans, AUC, EMP, loanprofit, p, 
   profit, profitPerLoan, statParityDiff)
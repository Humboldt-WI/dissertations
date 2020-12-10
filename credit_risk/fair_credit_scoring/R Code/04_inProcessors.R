# INPROCESSING DATA MODEL SELECTION

#setwd("C:/Users/Johannes/OneDrive/Dokumente/Humboldt-Universität/Msc WI/1_4. Sem/Master Thesis II/")
rm(list = ls());gc()
set.seed(0)
options(scipen=999)

# libraries
library(EMP)
library(pROC)
source("95_fairnessMetrics.R")

# read data
dval <- read.csv("taiwan_scaled_valid.csv")
dtest <- read.csv("taiwan_scaled_test.csv")
dtest_unscaled <- read.csv("/taiwan_orig_test.csv")

#-------------------------- PREJUDICE REMOVER ----------------------------------

dval_pred <- read.csv("taiwan_in_PRpredictions_valid.csv")
dtest_pred <- read.csv("taiwan_in_PRpredictions_test.csv")
dval_pred <- read.csv("taiwan_METApredictions_valid_07.csv")
dtest_pred <- read.csv("taiwan_METApredictions_test_07.csv")

#---- THRESHOLDING ----

# Find optimal cutoff based on validation set
empVals <- NULL
for (col in 1:ncol(dval_pred)){
  empVal <- empCreditScoring(dval_pred[,col], dval$TARGET)
  empVals <- unlist(c(empVals, empVal["EMPC"]))
}
bestPrediction <- dval_pred[, which(empVals == max(empVals))]
best_eta <- colnames(dval_pred)[which(empVals == max(empVals))]

# Define cutoff
EMP <- empCreditScoring(scores = bestPrediction, classes = dval$TARGET)
cutoff <- quantile(bestPrediction, EMP$EMPCfrac)
  
#---- TESTING ----

# Assess test restults
pred <- dtest_pred[,best_eta]
cutoff_label <- sapply(pred, function(x) ifelse(x>cutoff, 'Good', 'Bad'))

# Compute AUC
AUC <- as.numeric(roc(dtest$TARGET, as.numeric(pred))$auc)

# Compute EMP
EMP <- empCreditScoring(scores = pred, classes = dtest$TARGET)$EMPC
acceptedLoans <- length(pred[pred>cutoff])/length(pred)

# Compute Profit from Confusion Matrix (# means in comparison to base scenario = all get loan)
loanprofit <- NULL
for (i in 1:nrow(dtest)){
  class_label <- cutoff_label[i]
  true_label <- dtest$TARGET[i]
  if (class_label == "Bad" & true_label == "Bad"){
    #p = dtest_unscaled$CREDIT_AMNT[i]
    p = 0
  } else if (class_label == "Good" & true_label == "Bad"){
    p = -dtest_unscaled$CREDIT_AMNT[i] 
  } else if (class_label == "Good" & true_label == "Good"){
    p = dtest_unscaled$CREDIT_AMNT[i] * 0.2644
  }else if (class_label == "Bad" & true_label == "Good"){
    p = -dtest_unscaled$CREDIT_AMNT[i] * 0.2644
    #p = 0
  }
  loanprofit <- c(loanprofit, p)
}
profit <- sum(loanprofit)
profitPerLoan <- profit/nrow(dtest)

# fairness criteria average
statParityDiff <- statParDiff(sens.attr = dtest$AGE, target.attr = cutoff_label)
averageOddsDiff <- avgOddsDiff(sens.attr = dtest$AGE, target.attr = dtest$TARGET, predicted.attr = cutoff_label)
predParityDiff <- predParDiff(sens.attr = dtest$AGE, target.attr = dtest$TARGET, predicted.attr = cutoff_label)

cm <- confusionMatrix(data = as.factor(cutoff_label), reference = dtest$TARGET)
balAccuracy <- cm$byClass[['Balanced Accuracy']]

test_eval <- rbind(AUC, balAccuracy, EMP, acceptedLoans, profit, profitPerLoan, statParityDiff, averageOddsDiff, predParityDiff)

# Print results
test_eval
write.csv(test_eval, "5_finalResults/IN_Meta_Results.csv", row.names = T)




#-------------------------- ADVERSARIAL DEBIASING ------------------------------

dtest_pred <- read.csv("taiwan_advdebias_predictions.csv")

#---- TESTING ----

# Assess test restults
pred <- dtest_pred[,"labels"]
scores <- sapply(pred, function(x) ifelse(x==1, 1, 0))
cutoff_label <- sapply(pred, function(x) ifelse(x==1, 'Good', 'Bad'))

# Compute AUC
AUC <- as.numeric(roc(dtest$TARGET, as.numeric(scores))$auc)

# Compute EMP
EMP <- empCreditScoring(scores = scores, classes = dtest$TARGET)$EMPC
acceptedLoans <- length(pred[pred==1])/length(pred)

# Compute Profit from Confusion Matrix (# means in comparison to base scenario = all get loan)
loanprofit <- NULL
for (i in 1:nrow(dtest)){
  class_label <- cutoff_label[i]
  true_label <- dtest$TARGET[i]
  if (class_label == "Bad" & true_label == "Bad"){
    #p = dtest_unscaled$CREDIT_AMNT[i]
    p = 0
  } else if (class_label == "Good" & true_label == "Bad"){
    p = -dtest_unscaled$CREDIT_AMNT[i] 
  } else if (class_label == "Good" & true_label == "Good"){
    p = dtest_unscaled$CREDIT_AMNT[i] * 0.2644
  }else if (class_label == "Bad" & true_label == "Good"){
    p = -dtest_unscaled$CREDIT_AMNT[i] * 0.2644
    #p = 0
  }
  loanprofit <- c(loanprofit, p)
}
profit <- sum(loanprofit)
profitPerLoan <- profit/nrow(dtest)

# fairness criteria average
statParityDiff <- statParDiff(sens.attr = dtest$AGE, target.attr = cutoff_label)
averageOddsDiff <- avgOddsDiff(sens.attr = dtest$AGE, target.attr = dtest$TARGET, predicted.attr = cutoff_label)
predParityDiff <- predParDiff(sens.attr = dtest$AGE, target.attr = dtest$TARGET, predicted.attr = cutoff_label)

cm <- confusionMatrix(data = as.factor(cutoff_label), reference = dtest$TARGET)
balAccuracy <- cm$byClass[['Balanced Accuracy']]

test_eval <- rbind(AUC, balAccuracy, EMP, acceptedLoans, profit, profitPerLoan, statParityDiff, averageOddsDiff, predParityDiff)

# Print results
test_eval
write.csv(test_eval, "IN_AdvDebiasing_Results.csv", row.names = T)


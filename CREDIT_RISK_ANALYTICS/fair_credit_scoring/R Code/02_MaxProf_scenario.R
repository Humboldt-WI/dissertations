# MAX PROFIT

#setwd("C:/Users/Johannes/OneDrive/Dokumente/Humboldt-Universität/Msc WI/1_4. Sem/Master Thesis II/")
rm(list = ls());gc()
set.seed(0)
options(scipen=999)

# libraries
library(EMP)
library(pROC)
source("95_fairnessMetrics.R")

# read data
dtest_unscaled <- read.csv("taiwan_orig_test.csv")
dtest_unscaled <- subset(dtest_unscaled, select = c(CREDIT_AMNT,AGE, TARGET))

POST_results <- read.csv("taiwan_post_training_results_dtest.csv")

#---- TESTING ----
model.names <- c('glm', "svmLinear", "rf", "xgbTree", "nnet")
test_results <- NULL
for(m in model.names){
  # Assess test restults
  cutoff_label <- POST_results[,paste0(m, "_class")]
  #cutoff_label <- factor(as.character(cutoff_label), levels = c("Good", "Bad"))
  scores <- POST_results[,paste0(m, "_scores")]
  
  # Compute AUC
  AUC <- as.numeric(roc(dtest_unscaled$TARGET, as.numeric(scores))$auc)
  
  # Compute EMP
  EMP <- empCreditScoring(scores = scores, classes = dtest_unscaled$TARGET)$EMPC
  acceptedLoans <- length(cutoff_label[cutoff_label=="Good"])/length(cutoff_label)
  
  # Compute Profit from Confusion Matrix (# means in comparison to base scenario = all get loan)
  loanprofit <- NULL
  for (i in 1:nrow(dtest_unscaled)){
    class_label <- cutoff_label[i]
    true_label <- dtest_unscaled$TARGET[i]
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
  profitPerLoan <- profit/nrow(dtest_unscaled)
  
  # fairness criteria average
  statParityDiff <- statParDiff(sens.attr = dtest_unscaled$AGE, target.attr = cutoff_label)
  averageOddsDiff <- avgOddsDiff(sens.attr = dtest_unscaled$AGE, target.attr = dtest_unscaled$TARGET, predicted.attr = cutoff_label)
  predParityDiff <- predParDiff(sens.attr = dtest_unscaled$AGE, target.attr = dtest_unscaled$TARGET, predicted.attr = cutoff_label)
  
  cm <- confusionMatrix(data = as.factor(cutoff_label), reference = dtest_unscaled$TARGET)
  balAccuracy <- cm$byClass[['Balanced Accuracy']]
  
  test_eval <- rbind(AUC, balAccuracy, EMP, acceptedLoans, profit, profitPerLoan, statParityDiff, averageOddsDiff, predParityDiff)
  test_results <- cbind(test_results, test_eval)
  
}  
# Print results
colnames(test_results) <- c(model.names); test_results

write.csv(test_results, "MAX_PROFIT_Results.csv", row.names = T)

# EOP BY HAND

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

dval <- read.csv("taiwan_orig_valid.csv")

val_pred <- read.csv("taiwan_post_training_results_dval.csv")
val_pred <- cbind(val_pred, AGE = dval$AGE, TARGET = dval$TARGET)

test_pred <- read.csv("taiwan_post_training_results_dtest.csv")
test_pred <- cbind(test_pred, AGE = dtest_unscaled$AGE, TARGET = dtest_unscaled$TARGET, CREDIT_AMNT = dtest_unscaled$CREDIT_AMNT)


# find ROC curve
val_pred <- cbind(val_pred, AGE = dval$AGE, TARGET = dval$TARGET)

model.names <- c('glm', "svmLinear", "rf", "xgbTree", "nnet")
for (m in model.names){
  roc_valid_1 <- roc(val_pred$TARGET[val_pred$AGE==1], val_pred[val_pred$AGE==1,paste0(m, "_scores")])
  roc_valid_0 <- roc(val_pred$TARGET[val_pred$AGE==0], val_pred[val_pred$AGE==0,paste0(m, "_scores")])
  plot(roc_valid_0, col= "red", main = m)
  lines(roc_valid_1, col="green")
}
model.names <- c('glm', "svmLinear", "rf", "xgbTree", "nnet")
for (m in model.names){
  coordinates <- NULL
  for (t in seq(0.1,0.9,0.1)){
    EMP <- empCreditScoring(scores = sapply(val_pred[,paste0(m, "_scores")], function(x) ifelse(x<=t, 0, 1)), classes = val_pred$TARGET)$EMPC
    cutoff_label <- sapply(val_pred[val_pred$AGE==1,paste0(m, "_scores")], function(x) ifelse(x<=t, "Bad", "Good"))
    cm <- confusionMatrix(data = as.factor(cutoff_label), reference = val_pred$TARGET[val_pred$AGE==1])
    sens_1 <- cm$byClass[["Sensitivity"]]
    spec_1 <- cm$byClass[["Specificity"]]
    cutoff_label <- sapply(val_pred[val_pred$AGE==0,paste0(m, "_scores")], function(x) ifelse(x<=t, "Bad", "Good"))
    cm <- confusionMatrix(data = as.factor(cutoff_label), reference = val_pred$TARGET[val_pred$AGE==0])
    sens_0 <- cm$byClass[["Sensitivity"]]
    spec_0 <- cm$byClass[["Specificity"]]
    coors <- cbind(t, EMP, sens_0, sens_1, spec_0, spec_1)
    coordinates <- rbind(coordinates, coors)
  }
  #roc_valid_1 <- roc(val_pred$TARGET[val_pred$AGE==1], val_pred[val_pred$AGE==1,paste0(m, "_scores")])
  #roc_valid_0 <- roc(val_pred$TARGET[val_pred$AGE==0], val_pred[val_pred$AGE==0,paste0(m, "_scores")])
  #plot(roc_valid_0, col= "red", main = m)
  #lines(roc_valid_1, col="green")
}


for (m in model.names){
  # 0 is unprivileged group
  # Find threshold that optimizes the unpriviliged group via EMP
  pred_0 <- val_pred[val_pred$AGE==0,paste0(m, "_scores")]
  EMP <- empCreditScoring(scores = pred_0, classes = val_pred$TARGET[val_pred$AGE==0])
  assign(paste0('0_cutoff.', m), quantile(pred_0, EMP$EMPCfrac))
  cutoff_label <- sapply(val_pred[val_pred$AGE==0,paste0(m, "_scores")], function(x) ifelse(x<=quantile(pred_0, EMP$EMPCfrac), "Bad", "Good"))
  # get the sensitivity
  cm <- confusionMatrix(data = as.factor(cutoff_label), reference = val_pred$TARGET[val_pred$AGE==0])
  sens_0 <- cm$byClass[["Sensitivity"]]
  # find the threshold for the privileged group with the same sensitivity
  roc_curve <- roc(val_pred$TARGET[val_pred$AGE==1], val_pred[val_pred$AGE==1,paste0(m, "_scores")])
  my.coords <- coords(roc=roc_curve, x = "all", transpose = FALSE)
  assign(paste0('1_cutoff.', m),my.coords[which.min(abs(my.coords$sensitivity-sens_0)), ]$threshold)
}

# TEST RESULTS
test_results <- NULL
for(m in model.names){
  # Assess test restults
  cutoff_label_0 <- sapply(test_pred[test_pred$AGE==0, paste0(m, "_scores")], function(x) ifelse(x <= get(paste0("0_cutoff.",m)), "Bad", "Good"))
  cutoff_label_1 <- sapply(test_pred[test_pred$AGE==1, paste0(m, "_scores")], function(x) ifelse(x <= get(paste0("1_cutoff.",m)), "Bad", "Good"))
  cutoff_label <- c(cutoff_label_0, cutoff_label_1)
  test_label <- c(as.character(test_pred$TARGET[test_pred$AGE==0]), as.character(test_pred$TARGET[test_pred$AGE==1]))
  test_label <- as.factor(test_label)
  credit <- c(test_pred$CREDIT_AMNT[test_pred$AGE==0], test_pred$CREDIT_AMNT[test_pred$AGE==1])
  age <- c(rep(0,length(cutoff_label_0)), rep(1,length(cutoff_label_1)))
  
  acceptedLoans <- length(cutoff_label[cutoff_label=="Good"])/length(cutoff_label)
  
  # Compute Profit from Confusion Matrix (# means in comparison to base scenario = all get loan)
  loanprofit <- NULL
  for (i in 1:nrow(dtest_unscaled)){
    class_label <- cutoff_label[i]
    true_label <- test_label[i]
    if (class_label == "Bad" & true_label == "Bad"){
      #p = dtest_unscaled$CREDIT_AMNT[i]
      p = 0
    } else if (class_label == "Good" & true_label == "Bad"){
      p = -credit[i] 
    } else if (class_label == "Good" & true_label == "Good"){
      p = credit[i] * 0.2644
    }else if (class_label == "Bad" & true_label == "Good"){
      p = -credit[i] * 0.2644
      #p = 0
    }
    loanprofit <- c(loanprofit, p)
  }
  profit <- sum(loanprofit)
  profitPerLoan <- profit/length(test_label)
  
  # fairness criteria average
  statParityDiff <- statParDiff(sens.attr = age, target.attr = cutoff_label)
  averageOddsDiff <- avgOddsDiff(sens.attr = age, target.attr = test_label, predicted.attr = cutoff_label)
  predParityDiff <- predParDiff(sens.attr = age, target.attr = test_label, predicted.attr = cutoff_label)
  
  cm <- confusionMatrix(data = as.factor(cutoff_label), reference = test_label)
  balAccuracy <- cm$byClass[['Balanced Accuracy']]
  
  test_eval <- rbind(balAccuracy, acceptedLoans, profit, profitPerLoan, statParityDiff, averageOddsDiff, predParityDiff)
  test_results <- cbind(test_results, test_eval)
  
}  
# Print results
colnames(test_results) <- c(model.names); test_results

write.csv(test_results, "POST_EOP_Results.csv", row.names = T)

# POSTPROCESSING PROFIT EVALUATION

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

dval_unscaled <- read.csv("taiwan_orig_valid.csv")
dval_unscaled <- subset(dval_unscaled, select = c(AGE, TARGET))

dval_training_results <- read.csv("taiwan_post_training_results_dval.csv")
dtest_training_results <- read.csv("taiwan_post_training_results_dtest.csv")

# ---- PLATT SCALING PER GROUP ----
model.names <- c('glm', "svmLinear", "rf", "xgbTree", "nnet")
for (i in c(0,1)){
  dval_target <- dval_unscaled$TARGET[dval_unscaled$AGE==i]
  dval_scores <- dval_training_results[dval_unscaled$AGE==i,]
  dtest_scores <- dtest_training_results[dtest_unscaled$AGE==i,]
  dtest_subset <- dtest_unscaled[dtest_unscaled$AGE==i,]
  platt_scores <- NULL
  for (m in model.names){
    # train logistic model with Yval ~ Y^val --> model_val
    dataframe_valid <- data.frame(x = dval_scores[, paste0(m, "_scores")], y = dval_target)
    model_val <- glm(y~x,data = dataframe_valid,family = binomial)
    
    # determine optimal cutoff
    dataframe_valid <- dataframe_valid[-2]
    valid_scores <- predict(model_val, newdata = dataframe_valid, type = 'response')
    EMP <- empCreditScoring(scores = valid_scores, classes = dval_target)
    assign(paste0('cutoff.', m), quantile(valid_scores, EMP$EMPCfrac))
    
    # use model_val to predict ytest
    dataframe_test <- data.frame(x = dtest_scores[, paste0(m, "_scores")])
    test_score <- predict(model_val, newdata = dataframe_test, type = 'response')
    platt_scores <- cbind(platt_scores, test_score)
  }
  colnames(platt_scores) <- model.names
  assign(paste0("platt_scores_",i), cbind(platt_scores, dtest_subset))
}
platt_results <- rbind(platt_scores_0, platt_scores_1)


#---- TESTING ----

# Assess test restults
test_results <- NULL

for(i in model.names){
  
  pred <- platt_results[, i]
  cutoff <- get(paste0("cutoff.", i))
  cutoff_label <- sapply(pred, function(x) ifelse(x>cutoff, 'Good', 'Bad'))
  
  # Compute AUC
  AUC <- as.numeric(roc(platt_results$TARGET, as.numeric(pred))$auc)
  
  # Compute EMP
  EMP <- empCreditScoring(scores = pred, classes = platt_results$TARGET)$EMPC
  acceptedLoans <- length(pred[pred>cutoff])/length(pred)
  
  # Compute Profit from Confusion Matrix (# means in comparison to base scenario = all get loan)
  loanprofit <- NULL
  for (i in 1:nrow(platt_results)){
    class_label <- cutoff_label[i]
    true_label <- platt_results$TARGET[i]
    if (class_label == "Bad" & true_label == "Bad"){
      #p = dtest_unscaled$CREDIT_AMNT[i]
      p = 0
    } else if (class_label == "Good" & true_label == "Bad"){
      p = -platt_results$CREDIT_AMNT[i] 
    } else if (class_label == "Good" & true_label == "Good"){
      p = platt_results$CREDIT_AMNT[i] * 0.2644
    }else if (class_label == "Bad" & true_label == "Good"){
      p = -platt_results$CREDIT_AMNT[i] * 0.2644
      #p = 0
    }
    loanprofit <- c(loanprofit, p)
  }
  profit <- sum(loanprofit)
  profitPerLoan <- profit/nrow(platt_results)
  
  # fairness criteria average
  statParityDiff <- statParDiff(sens.attr = platt_results$AGE, target.attr = cutoff_label)
  averageOddsDiff <- avgOddsDiff(sens.attr = platt_results$AGE, target.attr = platt_results$TARGET, predicted.attr = cutoff_label)
  predParityDiff <- predParDiff(sens.attr = platt_results$AGE, target.attr = platt_results$TARGET, predicted.attr = cutoff_label)
  
  cm <- confusionMatrix(data = as.factor(cutoff_label), reference = platt_results$TARGET)
  balAccuracy <- cm$byClass[['Balanced Accuracy']]
  
  test_eval <- rbind(AUC, balAccuracy, EMP, acceptedLoans, profit, profitPerLoan, statParityDiff, averageOddsDiff, predParityDiff)
  test_results <- cbind(test_results, test_eval)
}

# Print results
colnames(test_results) <- c(model.names); test_results

write.csv(test_results, "POST_PlattScaling_Results.csv", row.names = T)





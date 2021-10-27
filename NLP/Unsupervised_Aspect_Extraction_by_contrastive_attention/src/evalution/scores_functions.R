# Functions used to calculate confusion matrix, per-class and macro scores
# This implementation is based on
# https://blog.revolutionanalytics.com/2016/03/com_class_eval_metrics_r.html

# Confusion matrix
get_cm <- function(y_true, y_pred) {
  y_true.factored <- factor(y_true, levels = unique(c(y_true, y_pred)))
  y_pred.factored <- factor(y_pred, levels = unique(c(y_true, y_pred)))
  return(as.matrix(table(Pred = y_pred.factored, True = y_true.factored)))
}

# Get per-class and macro precision, recall and F-score from the confusion matrix
get_scores_from_cm <- function(cm) {
  n <- sum(cm) # total number of observations
  nc <- nrow(cm) # number of classes
  diag <- diag(cm) # vector of true positives
  rowsums <- apply(cm, 1, sum) # number of ture observations per class
  colsums <- apply(cm, 2, sum) # number of predictions per class

  # Accuracy, precision, recall and F-score
  accuracy <- sum(diag) / n
  precision <- diag / rowsums
  recall <- diag / colsums
  f1 <- 2 * precision * recall / (precision + recall)

  # Create a new dataframe with class scores
  scores <- data.frame(precision, recall, f1)

  # Assume 0 in case of zero division
  scores[is.na(scores)] <- 0

  # Weighted and unweighted macro scores for precision, recall and F1
  unweighted_macro_scores <- apply(scores, 2, sum) / nc
  weighted_macro_score <- apply(scores * colsums, 2, sum) / n

  # Append weighted and unweighted macro scores to results dataframe
  scores_table <- rbind(scores, unweighted_macro_scores, weighted_macro_score)
  rownames(scores_table) <- c(rownames(scores), "unweighted", "weighted")

  # Append accuracy to dataframe
  scores_table <- cbind(accuracy = NA, scores_table)
  scores_table["unweighted", "accuracy"] <- accuracy

  return(scores_table)
}


# Get scores and confusion matrix
get_scores <- function(y_true, y_pred) {
  cm <- get_cm(y_true, y_pred)
  return(get_scores_from_cm(cm))
}

# Extract weighted scores from the result dataframe at given parameters.
get_weighted_scores_for_params <- function(testset, s = 100, i = 0, g = 0.03, attention = "rbf_attention") {
  subset <- df_all %>% filter(test_dataset == testset &
                                train_size == s &
                                iteration == i &
                                gamma == g &
                                attention_func == attention)

  y_true <- subset$y_true
  y_pred <- subset$y_pred
  scores <- get_scores(y_true, y_pred)
  weighted_precision <- scores["weighted", "precision"]
  weighted_recall <- scores["weighted", "recall"]
  weighted_f1 <- scores["weighted", "f1"]
  return(c(precision = weighted_precision, recall = weighted_recall, f1 = weighted_f1))
}

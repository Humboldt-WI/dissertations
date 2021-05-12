library(dplyr)
library(glue)
library(DBI)
library(RSQLite)

source("src/evalution/scores_functions.R")

sqlite_db_path <- "src/resources/sql/results.sqlite3"
con <- dbConnect(SQLite(), sqlite_db_path)

# Feth all results from the database
df_all <- dbReadTable(con, 'results')
dbDisconnect(con)

# Evaluate all datasets. This can also be restricted to a subset.
testsets <- unique(df_all$test_dataset)
# testsets <- c('semeval_lapt_2015_combined', 'citysearch')


# Main experiment
# Get the main results ( at 100%, rbf attention, first iteration
for (testset in testsets) {
  df <- df_all %>% filter(test_dataset == testset &
                            iteration == 0 &
                            train_size == 100 &
                            gamma == 0.03 &
                            attention_func == "rbf_attention")

  y_true <- df$y_true
  y_pred <- df$y_pred


  # Compute the confusion matrix
  cm <- get_cm(y_true, y_pred)
  print(cm)

  # Compute accuracy, precision, recall and F-score
  # as well as weighted and unweighted macro scores.
  scores <- get_scores_from_cm(cm)
  print(scores)
}

# Sub-experiment corpus size
# Plot the effect of the corpus size on the performance of CAt
for (testset in unique(testsets)) {
  df <- df_all %>% filter(test_dataset == testset)

  # Get all sample sizes and iterations from results and get the cartesian product
  train_sizes <- sort(unique(df$train_size))
  iterations <- unique(df$iteration)
  grid <- expand.grid(s = train_sizes, i = iterations)

  # Get the scores at each tuple (iteration, sample size)
  res <- mapply(get_weighted_scores_for_params,
                testset = list(testset),
                s = grid$s,
                i = grid$i,
                SIMPLIFY = TRUE)

  # Append scores to list of tuples
  grid$precision <- res["precision",]
  grid$recall <- res["recall",]

  # Compute average scores and standard deviation
  # for each iteration over different sample size.
  grid_grouped <- grid %>%
    group_by(s) %>%
    summarise(
      precision_mean = mean(precision),
      precision_sd = sd(precision),
      recall_mean = mean(recall),
      recall_sd = sd(recall),
    )

  # Change the output device for the plots, write plots to file
  filename <- glue("out/plots/plot-sample-size-{testset}.pdf")
  pdf(filename, height = 7, width = 12)

  # Plot results
  with(grid_grouped, {
    plot(x = 0, y = 0, type = "n", xlim = c(0, 20), ylim = c(0, 1),
         axes = FALSE, xlab = "% of training dataset", ylab = "Precision, Recall")
    axis(side = 2)
    axis(side = 1, at = seq(0, 20, 2), labels = seq(0, 100, 10))

    lines(precision_mean, type = "l", col = "blue")
    lines(precision_mean + precision_sd, type = "l", lty = "dashed", col = "skyblue")
    lines(precision_mean - precision_sd, type = "l", lty = "dashed", col = "skyblue")

    lines(recall_mean, type = "l", col = "red")
    lines(recall_mean + recall_sd, type = "l", lty = "dashed", col = "salmon")
    lines(recall_mean - recall_sd, type = "l", lty = "dashed", col = "salmon")
  })

  # Close the output device
  dev.off()

}

# Sub-experiment: influence of gamma
# Plot F1 for varying gamma
for (testset in testsets) {
  df <- df_all %>% filter(test_dataset == testset)
  attention_funcs <- unique(df$attention_func)
  gammas <- sort(unique(df$gamma))
  grid <- expand.grid(attention = attention_funcs, gamma = gammas)

  res <- mapply(get_weighted_scores_for_params, testset = list(testset), attention = grid$attention, g = grid$gamma, SIMPLIFY = TRUE)

  grid$f1 <- res["f1",]

  ylim_min <- round(min(subset(grid, attention == "rbf_attention")$f1) - 0.1, digits = 1)
  ylim_max <- round(max(subset(grid, attention == "rbf_attention")$f1) + 0.1, digits = 1)

  # Change the output device for the plots, write plots to file
  filename <- glue("out/plots/plot-gamma-{testset}.pdf")
  pdf(filename, height = 7, width = 12)

  plot(0, 0, type = "n", xlim = c(0, 0.05), ylim = c(ylim_min, ylim_max), xlab = "gamma", ylab = "F-score", axes = FALSE)
  axis(side = 2)
  axis(side = 1)

  lines(f1 ~ gamma, data = subset(grid, attention == "rbf_attention"), type = "l", col = "black")
  lines(f1 ~ gamma, data = subset(grid, attention == "no_attention"), type = "l", col = "blue")

  dev.off()
}

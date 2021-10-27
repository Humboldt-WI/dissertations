#### AUC function ####

AUC_binary <- function(score, wahrheit_binary) {
  #### Overhead ####
  # check packages
  require(magrittr)
  require(dplyr)
  
  # check function arguments: classes
  stopifnot(is.numeric(score))
  stopifnot(class(wahrheit_binary) %in% c("numeric", "integer", "logical"))
  
  # check function arguments: values
  stopifnot(length(score) == length(wahrheit_binary))
  stopifnot(all(wahrheit_binary %in% c(TRUE, FALSE, 1, 0)))
  
  #### Core of the function ####
  
  N = length(score)
  n_links <- sum(wahrheit_binary)
  n <- n_links * (N - n_links)
  
  n_pos <- n + 
    sum((rank(complex(real=score, imaginary=wahrheit_binary)) - N) * wahrheit_binary) + 
    (n_links - 1) * n_links / 2
  n_neg <- n + 
    sum((rank(complex(real=score, imaginary=-wahrheit_binary)) - N) * wahrheit_binary) + 
    (n_links - 1) * n_links / 2
  
  #(n_bis + 0.5 * n_bisbis) / n
  AUC <-  (n_pos + n_neg) / 2 / n
  
  return(AUC)
}

#### K-fold cross-validation: splitting only the existent links ####

k_fold_hiding <- function(edges_df, binary_target_varname, k = 10, FUN, ...) {
  
  #### Overhead ####
  # check packages
  require(magrittr)
  require(dplyr)
  require(rlang)
  require(mltools)
  
  # check function arguments: classes
  stopifnot(is.data.frame(edges_df))
  stopifnot(is.character(binary_target_varname))
  stopifnot(is.integer(k) | is.numeric((k)))
  stopifnot(rlang::is_function(FUN))
  
  # check function arguments: values
  stopifnot(binary_target_varname %in% names(edges_df))
  
  binary_target <- edges_df[[binary_target_varname]]
  
  stopifnot(class(binary_target) %in% c("numeric", "integer", "logical"))
  stopifnot(all(binary_target %in% c(TRUE, FALSE, 1, 0)))
  stopifnot(mod(k, 1) == 0 | k > 0)
  
  #### Core of the function ####
  
  # define folds for hiding
  # natural positive number - is an index of fold, on which the link=1
  # is hidden/masked as =0
  fold <- binary_target
  fold[fold == 1] <- folds(sum(binary_target), k)
  
  # meaningful operations
  binary_AUCs <- sapply(1:k, function(i) {

    edges_df_fold <- edges_df %>% 
      mutate(!!binary_target_varname := ifelse(fold == i, 0L, binary_target))

    res <- data.frame(score = eval(FUN)(edges_df_fold, ...),
                      target = binary_target) %>% 
      filter(fold %in% c(0, i))
    
    return(AUC_binary(res$score, res$target))
  })
  
  # formatting and reporting
  return(binary_AUCs)
}

# this script return data frame "auc" which includes average value of AUC across k=10 folds for the defined metrics. 
# input file "edges_df" includes all possible edges, with parameter "link" equal zero for non-existing, and 1 for existing links
 

#### Population  and distance metrics function 
val1 <- c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
val2 <- c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
val3 <- c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)

cobbdouglas <- data.frame(matrix(ncol = 11, nrow = 0))
x <- c("id", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9", "k10")
colnames(cobbdouglas) <- x

for (a1 in 1:length(val1))
{
  for (a2 in 1:length(val2))
  {
    for (a3 in 1:length(val3))
    {
      pop_dist_Cobb_Douglas <- function(edges_df,
                                        alpha1 = val1[[a1]], 
                                        alpha2 = val2[[a2]], 
                                        alpha3 = val3[[a3]]) {
        
        #### Overhead ####
        # check packages
        require(magrittr)
        require(dplyr)
        
        # check function arguments: classes
        #stopifnot(is.data.frame(edges_df))
        #stopifnot(is.numeric(alpha1))
        #stopifnot(is.numeric(alpha2))
        #stopifnot(is.numeric(alpha3))
        
        # meaningful operations
        metrics = with(edges_df,
                       (pmax(total_pop_from, total_pop_to)^alpha1) * 
                         (pmin(total_pop_from, total_pop_to)^alpha2) / 
                         (distance^alpha3))
        
        # formatting and reporting
        return(metrics)
      }
      
      pop_dist_metrics <- k_fold_hiding(edges_df, "link", k = 10, 
                                        FUN = pop_dist_Cobb_Douglas )
      pop_dist <-as.list(pop_dist_metrics)
      
      tmp <- data.frame(matrix(ncol = 11, nrow = 0))
      
      tmp <- data.frame( paste(val1[[a1]], val2[[a2]], val3[[a3]]), pop_dist)
      colnames(tmp) <- x
      
      cobbdouglas <- rbind(cobbdouglas, tmp)
    }
  }
}

cobbdouglas$id <-  gsub(" ", "_", cobbdouglas$id,  fixed = TRUE)

cobb_douglas_metrics <- cobbdouglas %>% 
  dplyr::transmute(id, mean = rowMeans(dplyr::select(., -id)))


#### Network metrics ####

metrics_names <- c("aa", "cn", "jc", "sl", "hdi", "hpi",
                   "lhn_local", "pa", "ra", "lp")
network_metrics_k_fold_AUCs <- lapply(metrics_names, function(method) {
  k_fold_hiding(edges_df, "link", k=10, FUN = edge_metrics, method)
})

names(network_metrics_k_fold_AUCs) <- metrics_names
rm(metrics_names)

network_metrics <- as.data.frame(network_metrics_k_fold_AUCs)

network_metrics <- as.data.frame(t(network_metrics))
network_metrics$id <- factor(row.names(network_metrics))
network_metrics <- network_metrics %>% 
  dplyr::select(id, everything())
colnames(network_metrics) <- x
network_metrics <- network_metrics %>% 
  dplyr::transmute(id, mean = rowMeans(dplyr::select(., -id)))

auc <- rbind(cobb_douglas_metrics, network_metrics)

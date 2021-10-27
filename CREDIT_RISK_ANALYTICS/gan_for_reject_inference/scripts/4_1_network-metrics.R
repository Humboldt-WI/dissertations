# All cores of metrics functions are borrowed directly from 'linkprediction' package
# linkprediction cannot be used as it has some unnecessary limitations and redundant functionality

#### Local network metrics ####

# Common Neighbours
cn <- function(graph) {
  # calculate common neighbours for all possible edges between vertices (nodes),
  # that participate in al least one link
  return(cocitation(graph))
}

# Adamic-Adar Index
aa <- function(graph) {
  return(igraph::similarity(graph, method = "invlogweighted"))
}

# Jaccard Index
jc <- function(graph) {
  return(igraph::similarity.jaccard(graph))
}

#Salton Index
sl <- function(graph) {
  deg <- igraph::degree(graph)
  cn <- igraph::cocitation(graph)
  return(cn / outer(deg, deg, function(x, y) sqrt(x * y))) # it is a matrix
}

# Hub Promoted Index
hpi <- function(graph) {
  deg <- igraph::degree(graph)
  cn <- igraph::cocitation(graph)
  return(cn / outer(deg, deg, pmin))
}

# Hub Depressed Index
hdi <- function(graph) {
  deg <- igraph::degree(graph)
  cn <- igraph::cocitation(graph)
  return(cn / outer(deg, deg, pmax))
}

# Leicht-Holme-Newman Index
lhn_local <- function(graph) {
  deg <- igraph::degree(graph)
  cn <- igraph::cocitation(graph)
  return(cn / outer(deg, deg))
}

# Preferential Attachment
pa <- function(graph) {
  deg <- igraph::degree(graph)
  return(outer(deg, deg))
}

# Resource Allocation
ra <- function(graph) {
  
  n <- igraph::vcount(graph)
  score <- matrix(integer(n^2), nrow = n)
  
  neighbors <- igraph::neighborhood(graph, 1)
  neighbors <- lapply(neighbors, function(x) x[-1])
  
  degrees <- igraph::degree(graph)
  for (k in seq(n)){
    tmp <- neighbors[[k]]
    l <- degrees[[k]]
    if (l > 1){
      for (i in 1:(l-1)){
        n1 <- tmp[i]
        for (j in (i+1):l){
          n2 <- tmp[j]
          score[n1, n2] <- score[n1, n2] + 1 / l
          score[n2, n1] <- score[n2, n1] + 1 / l
        }
      }
    }
  }
  
  return(score) # that is a matrix
}

# Local Path Index
lp <- function(graph, eps = 0.01) {
  A <- igraph::get.adjacency(graph)
  score <- A %*% A
  score <- score + score %*% A * eps
  return(as.matrix(score))
}

edge_metrics <- function(edges_df, method, ...){
  
  # Overhead checks
  edges_df_overhead_checks <- function(edges_df) {
    
    # check packages
    require(magrittr)
    require(dplyr)
    require(igraph)
    require(plyr)
    require(reshape2)
    
    # check function arguments: classes
    stopifnot(is.data.frame(edges_df))
    
    # check function arguments: values
    stopifnot(all(c("from", "to", "link") %in% colnames(edges_df)))
    stopifnot(all(names(edges_df[1:2]) == c("from", "to")) | 
                all(names(edges_df[1:2]) == c("to", "from"))) 
    # edge vertices ids should be in the first two columns
    
    return(TRUE)
  }
  edges_df_overhead_checks(edges_df)
  
  stopifnot(method %in% c("aa", "cn", "jc", "sl", 
                          "hdi", "hpi",
                          "lhn_local", "pa", "ra", "lp"))
  
  # select all edges that are links in a graph 
  # with possibly many separate connected subgraphs: 
  # outcast edges will not be a problem
  graph <- graph_from_data_frame(d = edges_df %>% filter(link == 1), 
                                 directed = F)
  
  # Find score
  score_mat <- do.call(method, list(graph = graph, ...))
  
  # Format
  links_df <- melt(score_mat, 
                   varnames = c("from", "to")) %>% 
    dplyr::rename(score = value) %>%
    mutate(from = as.character(from), to = as.character(to))
  # append cn score to inital dataframe (to preserve order) and
  # replace NA with 0-score
  edges_df %<>% 
    left_join(links_df, by = c("from", "to")) %>% 
    mutate(score = if_else(is.na(score), 0, score))
  
  # Cleaning/garbage collection
  # detach("package:reshape2", unload = TRUE)
  # detach("package:plyr", unload = TRUE)
  
  # Return
  return(edges_df$score)
}

# testing:
# tmp <- edge_metrics(edges_df = edges_df, "cn")

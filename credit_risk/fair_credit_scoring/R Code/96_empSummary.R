# EMP summary

library(EMP)

#Cost function for trainControl
creditSummary <- function (data,
                           lev = NULL, 
                           model = NULL) {
  lvls <- levels(data$obs)
  if (length(lvls) > 2) 
    stop(paste("Your outcome has", length(lvls), 
               "levels. The assignmentSummary() function isn't appropriate."))
  if (!all(levels(data[, "pred"]) == lvls)) 
    stop("levels of observed and predicted data do not match")

  out <- EMP::empCreditScoring(scores = data$Good, classes = data$obs, ROI = 0.26, p0 = 0.49, p1 = 0.1)
  out <- out$EMPC        
  names(out) <- "EMP"#Metric name
  out
}
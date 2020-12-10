# Statistical metrics

statParDiff <- function(sens.attr = df$AGE, target.attr = df$TARGET){
  sens.var <- as.factor(sens.attr)
  target.var <- as.factor(target.attr)
  sens.lvls <- levels(sens.var)
  target.lvls <- levels(target.var)
  
  data <- cbind(sens.var, target.var)
  
  total.count <- nrow(data)
  target1.count <- nrow(data[target.var==target.lvls[1],])
  
  
  p_1 <- (nrow(data[sens.var == sens.lvls[1] & target.var==target.lvls[1],])/target1.count) * 
    (target1.count/total.count) / (nrow(data[sens.var == sens.lvls[1],])/total.count)
  p_2 <- (nrow(data[sens.var == sens.lvls[2] & target.var==target.lvls[1],])/target1.count) * 
    (target1.count/total.count) / (nrow(data[sens.var == sens.lvls[2],])/total.count)
  return (p_1-p_2)
}

avgOddsDiff <- function(sens.attr = df$AGE, target.attr = df$TARGET, predicted.attr = df$class){
  sens.attr <- as.factor(sens.attr)
  data = data.frame(sens.attr, target.attr, predicted.attr, stringsAsFactors = T)
  colnames(data) <- c("sens", "target", "pred")

  data_un <- data[data[, "sens"]==levels(data[, "sens"])[1],]
  FN_un <- nrow(data_un[data_un[,"target"] == "Bad" & data_un[,"pred"]=="Good",])
  FP_un <- nrow(data_un[data_un[,"target"] == "Good" & data_un[,"pred"]=="Bad",])
  TP_un <- nrow(data_un[data_un[,"target"] == "Bad" & data_un[,"pred"]=="Bad",])
  TN_un <- nrow(data_un[data_un[,"target"] == "Good" & data_un[,"pred"]=="Good",])
  FPR_un <- FP_un/(TN_un+FP_un)
  TPR_un <- TP_un/(TP_un+FN_un)
  
  data_priv <- data[data[, "sens"]==levels(data[, "sens"])[2],]
  FN_priv <- nrow(data_priv[data_priv[,"target"] == "Bad" & data_priv[,"pred"]=="Good",])
  FP_priv <- nrow(data_priv[data_priv[,"target"] == "Good" & data_priv[,"pred"]=="Bad",])
  TP_priv <- nrow(data_priv[data_priv[,"target"] == "Bad" & data_priv[,"pred"]=="Bad",])
  TN_priv <- nrow(data_priv[data_priv[,"target"] == "Good" & data_priv[,"pred"]=="Good",])
  FPR_priv <- FP_priv/(TN_priv+FP_priv)
  TPR_priv <- TP_priv/(TP_priv+FN_priv)
  
  
  return (((FPR_un-FPR_priv)+(TPR_un-TPR_priv))/2)
}


predParDiff <- function(sens.attr = df$AGE, target.attr = df$TARGET, predicted.attr = df$class){
  sens.attr <- as.factor(sens.attr)
  data = data.frame(sens.attr, target.attr, predicted.attr, stringsAsFactors = T)
  colnames(data) <- c("sens", "target", "pred")
  
  data_un <- data[data[, "sens"]==levels(data[, "sens"])[1],]
  pp_un <- nrow(data_un[data_un[,"target"] == "Good" & data_un[,"pred"]=="Good",])/nrow(data_un[data_un[,"pred"] == "Good",])
  
  data_priv <- data[data[, "sens"]==levels(data[, "sens"])[2],]
  pp_priv <- nrow(data_priv[data_priv[,"target"] == "Good" & data_priv[,"pred"]=="Good",])/nrow(data_priv[data_priv[,"pred"] == "Good",])
  
  return (pp_un-pp_priv)
}


tab.ratio <- function(data, sens.attr.name){
  t <- table(data[, sens.attr.name], data$TARGET)
  return(list(t,paste("The unpriviliged group (A=0) has", round(t[1,2]/(t[1,1]+t[1,2])*100), "% Good Credit Card Owners. The priviliged group (A=1) has",
               round(t[2,2]/(t[2,1]+t[2,2])*100), "% Good Credit Card Owners.")))
}
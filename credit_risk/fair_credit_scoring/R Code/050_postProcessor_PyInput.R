# POSTPROCESSED DATA - PREDICTIONS

#setwd("C:/Users/Johannes/OneDrive/Dokumente/Humboldt-Universität/Msc WI/1_4. Sem/Master Thesis II/")
rm(list = ls());gc()
set.seed(0)
options(scipen=999)

# - load packages

packages <- c("caret", "doParallel", "kernlab", "randomForest", "nnet", 
              "xgboost", "foreach", "e1071", "pROC", "EMP")
sapply(packages, require, character.only = TRUE)

#Use parallel computing
nrOfCores  <- detectCores()-1
registerDoParallel(cores = nrOfCores)
message(paste("\n Registered number of cores:\n",nrOfCores,"\n"))

rm(packages, nrOfCores)

# - read data set (see Fairness Definitions Explained)

dtest <- read.csv("taiwan_scaled_test.csv")
dval <- read.csv("taiwan_scaled_valid.csv")
dtrain <- read.csv("taiwan_scaled_train.csv")

# check fairness in training set
source("95_fairnessMetrics.R")

#set trainControl for caret
source("96_empSummary.R")
model.control <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  verboseIter = T,
  allowParallel = TRUE,
  summaryFunction = creditSummary, #specific FAIR summary metric
  returnData = FALSE #FALSE, to reduce straining memomry 
)

#---- GRID SEARCH ----

#Specifications for rf
param.rf <- expand.grid(mtry = seq(5, 15, by = 5))
args.rf <- list(ntree = 1000)

#Specifications for nnet           
param.nnet <- expand.grid(decay = seq(0.1, 2, by =  0.1),
                          size = seq(10, 30, by = 1))
args.nnet <- list(maxit = 100, trace = FALSE)

#Specifications for xgbTree
param.xgbTree     <- expand.grid(
  nrounds = c(100, 200),
  max_depth = seq(8, 20, by = 2), 
  gamma = 0,
  eta = 0.1,
  colsample_bytree = seq(0.6, 1, by = 0.2),
  min_child_weight = c(0.5, 1, 3),
  subsample = seq(0.4, 0.8, by = 0.2)
)

args.xgbTree <- list()

#Specifications for svmLinear      
param.svmLinear <- expand.grid(C = seq(0.1, 10000, by =  1000))
args.svmLinear <- list()

#Specifications for glmnet
param.glm <- NULL
args.glm <- list(family = "binomial")

# Create vector of model names to call parameter grid in for-loop
model.names <- c(
  "glm",
  "svmLinear", 
  "rf", 
  "xgbTree",
  "nnet"
)


#---- TRAINING ----

# Train models and save result to model."name"
for(i in model.names) {
  print(i)
  grid <- get(paste("param.", i, sep = ""))
  
  args.train <- list(TARGET~., 
                     data = dtrain,  
                     method = i, 
                     tuneGrid  = grid,
                     metric    = "EMP", #needs to be changed depeding on cost function
                     trControl = model.control)
  
  args.model <- c(args.train
                  , get(paste("args.", i, sep = ""))
  )
  
  assign(
    paste("model.", i, sep = ""),
    do.call(train, args.model)
  )
  
  print(paste("Model", i, "finished training:", Sys.time(), sep = " "))
}

for (i in model.names){rm(list=c(paste0('args.',i), paste0('param.',i)))};gc()
rm(args.model, args.train, model.control)

#---- THRESHOLDING ----

# Find optimal cutoff based on validation set
for(i in model.names){
  # Define cutoff
  pred <- predict(get(paste("model.", i, sep = "")), newdata = dval, type = 'prob')$Good
  EMP <- empCreditScoring(scores = pred, classes = dval$TARGET)
  assign(paste0('cutoff.', i), quantile(pred, EMP$EMPCfrac))
}

#---- TRAINING RESULTS ----
data.names <- c("dval", "dtest")

for (data in data.names){
  model_prediction <- NULL
  cnames <- NULL
  for(i in model.names){
    
    pred <- predict(get(paste0("model.", i)), newdata = get(data), type = 'prob')$Good
    cutoff <- get(paste0("cutoff.", i))
    cutoff_label <- sapply(pred, function(x) ifelse(x>cutoff, 'Good', 'Bad'))
    
    model_prediction <- cbind(model_prediction, pred, cutoff_label)
    cnames <- c(cnames, c(paste0(i, "_scores"), paste0(i, "_class")))
  }
  colnames(model_prediction) <- cnames
  
  write.csv(model_prediction, paste0("taiwan_post_training_results_", data, ".csv"), row.names = F)
  
}



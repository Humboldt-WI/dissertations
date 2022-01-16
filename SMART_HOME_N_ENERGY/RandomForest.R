# HUMBOLDT-UNIVERSITAET ZU BERLIN   
# CHAIR OF INFORMATION SYSTEMS   
# Name:             Hee-Eun Lee
# Master's thesis:  Understanding User Perception and Intention to Use Smart Home for Energy Efficiency using Structural Equation Modeling and Machine Learning###
# 1st Examiner:     Prof. Dr. Lessmann
# 2nd Examiner:     PD. Dr. Fabian
# Supervisor:       Dr. Alona Zharova
#------------------------------------------------------------------------------------------------------------------------------------

# -- MACHINE LEARNING - RANDOM FOREST ALGORITHM --

# Load necessary libraries
if (!require("pacman")) install.packages("pacman")
pacman::p_load(randomForest, caret, ggplot2,dplyr)

# Set path to data set
cleanedDF_Path <- "C:\\Users\\heeeun\\Dropbox\\Uni\\Masterarbeit\\Code\\dfexcl2.csv"

# Statistically significant results from SEM are used as input for random forest algorithm
cleanedDF <-  read.csv(cleanedDF_Path, sep = ";")
summary(cleanedDF)

# Observed intentions BI1 to BI4 are combined by averaging the 7-point Likert scale values to the new column "BI" 
# Create new data frame "outputCleanedDF" including predictors (=indicators from SEM) and combined result for BI
cleanedDF$BI <- rowMeans(cleanedDF[,c("BI1","BI2","BI3","BI4")])
outputCleanedDF <- subset(cleanedDF, select = -c(BI1,BI2,BI3,BI4))


# --- Random Forest Analysis

# Set seed to not impact end result by randomization 
seed <- 100
set.seed(seed)

# Split data into train (70%) and validation set (30%)
train <- sample(nrow(outputCleanedDF), 0.7*nrow(outputCleanedDF), replace = FALSE)
TrainSet <- outputCleanedDF[train,]
ValidSet <- outputCleanedDF[-train,]
print(paste("Train-Set dimensions: ", nrow(TrainSet), ncol(TrainSet)))
print(paste("Validation-Set dimensions: ", nrow(ValidSet), ncol(ValidSet)))

# Extension of caret library to optimize parameters mtry and nodesize of the random forest algorithm
# Application of grid search to find optimal parameter settings and training of random forest on training set
set.seed(seed)
customRF <- list(type = "Regression", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "nodesize"), class = rep("numeric", 2), label = c("mtry", "nodesize "))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, mtry = param$mtry, nodesize=param$nodesize, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
   predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
   predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

# Training for random forest regression
# Train model for optimized parameters
control <- trainControl(method="repeatedcv", number=10, repeats=3)
tunegrid <- expand.grid(.mtry=c(1:8), .nodesize=c(1,3,5,10))
metric <- "RMSE"
set.seed(seed)
custom <- train(BI~., data=outputCleanedDF, method=customRF, metric=metric, tuneGrid=tunegrid, trControl=control, ntree=1000, importance = TRUE)
summary(custom)
print(custom)
plot(custom)
print(custom$finalModel)
plot(custom$finalModel)

# Using the trained random forest regression, data from the validation set is used to check for over-fitting and validity
pred <- predict(custom$finalModel, newdata = ValidSet)
postResample(pred,ValidSet$BI)

# Variable importance for regression, higher values indicate higher importance
importantVar <- varImp(custom)
print(importantVar)
ggplot(importantVar)

# Using the obtained optimized parameters, a final random forest regression algorithm is trained
# Train and summarize model 
set.seed(seed)
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# create standalone model using all training data
set.seed(seed)
finalModel <- randomForest(BI~., TrainSet,trControl=control, mtry=1, ntree=750, nodesize=3) # mtry and nodesize from grid optimization
print(finalModel)
# make predictions on "new data" using the final model (validation)
final_predictions <- predict(finalModel, ValidSet)

# Error over n-trees - plot
plot(finalModel)

# Results
print(finalModel)
print(postResample(final_predictions, ValidSet$BI))

# Compute importance of predictors
importantVar <- varImp(finalModel)
varImpPlot(finalModel,n.var=10)

importantVar$latentVariable <- substr(row.names(importantVar),1,2)
latVar_Imp <- importantVar %>%
	group_by(latentVariable) %>%
	summarise(Mean_Importance = mean(Overall))
latVar_Imp <- latVar_Imp[order(latVar_Imp$Mean_Importance),]
#lat Var_Imp[which(latVar_Imp$latentVariable == "PB"),"latentVariable"] <- "PBC"
barplot(latVar_Imp$Mean_Importance,main="Average IncNodePurity", horiz=TRUE, xlab="IncNodePurity",names.arg=latVar_Imp$latentVariable,xlim=c(0,30))


# --- Apply random forest to train the model again but using the average of latent variables as input
# Create average data frame
averageDF <- data.frame(outputCleanedDF)
#AT
averageDF$AT <- rowMeans(averageDF[,c("AT4","AT5","AT7")])
averageDF <- subset(averageDF, select = -c(AT4,AT5,AT7))
#SN
averageDF$SN <- rowMeans(averageDF[,c("SN1","SN2","SN3")])
averageDF <- subset(averageDF, select = -c(SN1,SN2,SN3))
#PBC
#averageDF$PBC <- rowMeans(averageDF[,c("PBC1","PBC4")])
#averageDF <- subset(averageDF, select = -c(PBC1,PBC4))
#AC
averageDF$AC <- rowMeans(averageDF[,c("AC1","AC2","AC3")])
averageDF <- subset(averageDF, select = -c(AC1,AC2,AC3))
#AR
averageDF$AR <- rowMeans(averageDF[,c("AR1","AR2","AR3")])
averageDF <- subset(averageDF, select = -c(AR1,AR2,AR3))
#PN
averageDF$PN <- rowMeans(averageDF[,c("PN1","PN2","PN3")])
averageDF <- subset(averageDF, select = -c(PN1,PN2,PN3))
#PB
averageDF$PB <- rowMeans(averageDF[,c("PB1","PB3")])
averageDF <- subset(averageDF, select = -c(PB1,PB3))

train <- sample(nrow(averageDF), 0.7*nrow(averageDF), replace = FALSE)
TrainSet <- averageDF[train,]
ValidSet <- averageDF[-train,]
print(paste("Train-Set dimensions: ", nrow(TrainSet), ncol(TrainSet)))
print(paste("Validation-Set dimensions: ", nrow(ValidSet), ncol(ValidSet)))

# train model
control <- trainControl(method="repeatedcv", number=10, repeats=3)
tunegrid <- expand.grid(.mtry=c(1:5), .nodesize=c(1,3,5,10))
metric <- "RMSE"
set.seed(seed)
custom <- train(BI~., data=averageDF, method=customRF, metric=metric, tuneGrid=tunegrid, trControl=control, ntree=750, importance = TRUE)
summary(custom)
print(custom)
plot(custom)
print(custom$finalModel)

# train and summarize model 
set.seed(seed)
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# create standalone model using all training data
set.seed(seed)
finalModel <- randomForest(BI~., TrainSet,trControl=control, mtry=1, ntree=750, nodesize=10)
print(finalModel)
# make a predictions on "new data" using the final model
final_predictions <- predict(finalModel, ValidSet)

print(finalModel)
print(postResample(final_predictions, ValidSet$BI))

importantVar <- varImp(finalModel)
varImpPlot(finalModel,n.var=6)

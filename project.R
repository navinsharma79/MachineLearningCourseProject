# https://stackoverflow.com/questions/23075506/how-to-improve-randomforest-performance

#training$classe = factor(training$classe)

set.seed(42)
library(caret)
library(rpart)
library(gbm)
library(forecast)
library(readr)
library(randomForest)

# load the data
dataset <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
validation <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")

# convert the type of exercise to a factor
dataset$classe = factor(dataset$classe)

# create training and testing sets
# the testing set will be used to estimate error rate
inTrain = createDataPartition(y=dataset$classe, p=0.75, list=FALSE)
training = dataset[inTrain,]
testing = dataset[-inTrain,]


NAcolumns <- colSums(is.na(training))/length(training$X)
NAcolumnsNames <- names(NAcolumns[NAcolumns > 0])

nonEmptyColumns <- colSums((training=="")/length(training$roll_belt))
nonEmptyColumnsNames <- names(nonEmptyColumns[nonEmptyColumns>0])

firstSeven <- names(training[,1:7])

training <- training[, !names(training) %in% NAcolumnsNames]
training <- training[, !names(training) %in% nonEmptyColumnsNames]
training <- training[, !names(training) %in% firstSeven]

# check for near zero variables
nsv <- nearZeroVar(training)

library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv", number = 3, allowParallel = TRUE)
ntrees <- 200
sample <- 1000

#rfFit <- train(classe~.,data=training,method="rf",trControl=fitControl,ntree=ntrees,sampsize=sample)
rfFit <- train(classe~.,data=training,method="rf",trControl=fitControl,ntree=ntrees)

stopCluster(cluster)
registerDoSEQ()

testing <- testing[, !names(testing) %in% NAcolumnsNames]
testing <- testing[, !names(testing) %in% nonEmptyColumnsNames]
testing <- testing[, !names(testing) %in% firstSeven]

pred <- predict(rfFit, newdata = testing)
confusionMatrix(pred,as.factor(testing$classe))

validation <- validation[, !names(validation) %in% NAcolumnsNames]
validation <- validation[, !names(validation) %in% nonEmptyColumnsNames]
validation <- validation[, !names(validation) %in% firstSeven]

pred2 <- predict(rfFit, newdata = validation)
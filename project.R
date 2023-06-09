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
testing <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")

# convert the type of exercise to a factor
dataset$classe = factor(dataset$classe)

# create training and validation sets
# the validation set will be used to estimate error rate
inTrain = createDataPartition(y=dataset$classe, p=0.75, list=FALSE)
training = dataset[inTrain,]
validation = dataset[-inTrain,]


NAcolumns <- colSums(is.na(training))/length(training$X)
NAcolumnsNames <- names(NAcolumns[NAcolumns > 0])

nonEmptyColumns <- colSums((training=="")/length(training$roll_belt))
nonEmptyColumnsNames <- names(nonEmptyColumns[nonEmptyColumns>0])

firstSeven <- names(training[,1:7])

training <- training[, !names(training) %in% NAcolumnsNames]
training <- training[, !names(training) %in% nonEmptyColumnsNames]
training <- training[, !names(training) %in% firstSeven]

# check for near zero variables
nzv <- nearZeroVar(training)

# parallel processing
library(parallel)
library(doParallel)


# 5 trees
cluster <- makeCluster(detectCores() - 1) # leave one core for others
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv", number = 5, allowParallel = TRUE)
ntrees <- 5
rfFit5Trees <- train(classe~.,data=training,method="rf",trControl=fitControl,ntree=ntrees)

stopCluster(cluster)
registerDoSEQ()

# 5 trees bootstrap
cluster <- makeCluster(detectCores() - 1) # leave one core for others
registerDoParallel(cluster)

fitControl <- trainControl(method = "boot", allowParallel = TRUE)
ntrees <- 5
rfFit5TreesBoot <- train(classe~.,data=training,method="rf",trControl=fitControl,ntree=ntrees)

stopCluster(cluster)
registerDoSEQ()


# 10 trees
cluster <- makeCluster(detectCores() - 1) # leave one core for others
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv", number = 5, allowParallel = TRUE)
ntrees <- 10
rfFit10Trees <- train(classe~.,data=training,method="rf",trControl=fitControl,ntree=ntrees)

stopCluster(cluster)
registerDoSEQ()

# 50 trees
cluster <- makeCluster(detectCores() - 1) # leave one core for others
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv", number = 5, allowParallel = TRUE)
ntrees <- 50
rfFit50Trees <- train(classe~.,data=training,method="rf",trControl=fitControl,ntree=ntrees)

stopCluster(cluster)
registerDoSEQ()

# 100 trees
cluster <- makeCluster(detectCores() - 1) # leave one core for others
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv", number = 5, allowParallel = TRUE)
ntrees <- 100
rfFit100Trees <- train(classe~.,data=training,method="rf",trControl=fitControl,ntree=ntrees)

stopCluster(cluster)
registerDoSEQ()

# 200 trees
cluster <- makeCluster(detectCores() - 1) # leave one core for others
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv", number = 5, allowParallel = TRUE)
ntrees <- 200
rfFit200Trees <- train(classe~.,data=training,method="rf",trControl=fitControl,ntree=ntrees)

stopCluster(cluster)
registerDoSEQ()

validation <- validation[, !names(validation) %in% NAcolumnsNames]
validation <- validation[, !names(validation) %in% nonEmptyColumnsNames]
validation <- validation[, !names(validation) %in% firstSeven]

pred5 <- predict(rfFit5Trees, newdata = validation)
confusionMatrix(pred5,validation$classe)

pred10 <- predict(rfFit10Trees, newdata = validation)
confusionMatrix(pred10,as.factor(validation$classe))

pred50 <- predict(rfFit50Trees, newdata = validation)
confusionMatrix(pred50,as.factor(validation$classe))

pred100 <- predict(rfFit100Trees, newdata = validation)
confusionMatrix(pred100,as.factor(validation$classe))

pred200 <- predict(rfFit200Trees, newdata = validation)
confusionMatrix(pred200,as.factor(validation$classe))

treeAccuracy <- data.frame(
  Trees = c(5,10,50,100,200),
  Accuracy = c(confusionMatrix(pred5,validation$classe)$overall[1],
               confusionMatrix(pred10,validation$classe)$overall[1],
               confusionMatrix(pred50,validation$classe)$overall[1],
               confusionMatrix(pred100,validation$classe)$overall[1],
               confusionMatrix(pred200,validation$classe)$overall[1])
)

testing <- testing[, !names(testing) %in% NAcolumnsNames]
testing <- testing[, !names(testing) %in% nonEmptyColumnsNames]
testing <- testing[, !names(testing) %in% firstSeven]

pred2 <- predict(rfFit, newdata = testing)
#source("dataPrepare.R")

#Normalize the data*************************************************************
normalit<-function(m){
     (m - min(m))/(max(m)-min(m))
}

Target <- data.final$Target
data.ind <- apply(data.final[,1:length(data.final)-1],2,normalit)
data.norm <- cbind.data.frame(data.ind,Target)

#Feature Selection**************************************************************
library(caret)
library(plyr)
set.seed(915)

data.norm$Target <- mapvalues(data.norm$Target, 
                              from = c("1","2","3","4","5","6"),
                              to = c("C1","C2","C3","C4","C5","C6"))

control <- trainControl(method="repeatedcv", number=5, repeats=3,
                        savePredictions = TRUE)
model <- train(Target~., data=data.norm, method="svmPoly",
               preProcess="scale",trControl=control)

importance <- varImp(model, scale=TRUE)

print(importance)
plot(importance, main = "Important factors")

#options(gsubfn.engine = "R")
library(caret)
#library(fscaret)
library(plyr)

set.seed(915)

#Normalize the data*************************************************************
normalit<-function(m){
     (m - min(m))/(max(m)-min(m))
}

Target <- data.final$Target
data.ind <- apply(data.final[,1:length(data.final)-1],2,normalit)
data.norm <- cbind.data.frame(data.ind,Target)

data.norm$Target <- mapvalues(data.norm$Target, 
                              from = c("1","2","3","4","5","6"),
                              to = c("C1","C2","C3","C4","C5","C6"))


#Ensemble Feature Selection*****************************************************

data.size <- nrow(data.norm)
# timeSlice <- createTimeSlices(data.norm$Target, initialWindow = 0.5*data.size,
#                               horizon = 0.1*data.size,fixedWindow = TRUE,
#                               skip = 0.04*data.size)
# 
# 
# trainDF <- data.norm[timeSlice$train[[1]],]
# testDF  <- data.norm[timeSlice$test[[1]],]

spliter <- as.integer(0.8*nrow(data.norm))
data.validation <- data.norm[(spliter + 1):nrow(data.norm),]

# control <- trainControl(method="timeslice",initialWindow = 0.7*data.size, 
#                         horizon = 0.2*data.size,fixedWindow = TRUE,
#                         savePredictions = TRUE)

control <- trainControl(method="repeatedcv", number=5, repeats=3,
                        savePredictions = TRUE)

model <- train(Target~., data=data.norm, method="svmPoly", preProcess="scale",
               trControl=control)

importance <- varImp(model, scale=TRUE)

print(importance)
plot(importance, main = "Important factors")


getImp <- function(importance){
     imp = importance$importance
     imp = transform(imp, sum = rowSums(imp))
     imp.x = row.names(imp)
     imp.y = imp$sum
     imp_data = cbind.data.frame(imp.x, imp.y)
     imp_data = imp_data[order(imp_data$imp.y),]
     return(imp_data)
}

imp = importance$importance
imp = transform(imp, sum = rowSums(imp))

imp.x = row.names(imp)
imp.y = imp$sum/6

imp_data = cbind.data.frame(imp.x, imp.y)
imp_data = imp_data[order(imp_data$imp.y),]

par(mai = c(0.5,2.3,0.2,0.8))
barplot(imp_data$imp.y, main = "Importatnt Factors", 
        horiz = TRUE, names.arg = imp_data$imp.x, las=1,
        cex.axis = 1.28, cex.names = 1.28, cex.main = 1.5)


#part 2: Random forest
control.rf <- trainControl(method="repeatedcv", number=5, repeats=3,
                        savePredictions = TRUE)

model.rf <- train(Target~., data=data.norm, method="rf", preProcess="scale",
               trControl=control)


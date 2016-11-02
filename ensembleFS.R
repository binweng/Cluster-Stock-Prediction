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

#Part1: Support Vector Machine**************************************************

control.svm <- trainControl(method="repeatedcv", number=5, repeats=3,
                        savePredictions = TRUE)

model.svm <- train(Target~., data=data.norm, method="svmPoly", preProcess="scale",
               trControl=control.svm)

importance.svm <- varImp(model.svm, scale=TRUE)

#print(importance)
plot(importance.svm, main = "Important factors")

imp.svm = importance.svm$importance
imp.svm = transform(imp.svm, sum = rowSums(imp.svm))

imp.svm.x = row.names(imp.svm)
imp.svm.y = imp.svm$sum/6

imp_data.svm = cbind.data.frame(imp.svm.x, imp.svm.y)
imp_data.svm = imp_data.svm[order(imp_data.svm$imp.svm.y),]

par(mai = c(0.5,2.3,0.2,0.8))
barplot(imp_data.svm$imp.svm.y, main = "Importatnt Factors", 
        horiz = TRUE, names.arg = imp_data.svm$imp.svm.x, las=1,
        cex.axis = 1.28, cex.names = 1.28, cex.main = 1.5)


#part 2: Random forest**********************************************************
control.rf <- trainControl(method="repeatedcv", number=5, repeats=3,
                        savePredictions = TRUE)

model.rf <- train(Target~., data=data.norm, method="rf", preProcess="scale",
               trControl=control.rf)

importance.rf <- varImp(model.rf, scale=TRUE)

# getImp <- function(importance){
#      imp = importance$importance
#      imp = transform(imp, sum = rowSums(imp))
#      imp.x = row.names(imp)
#      imp.y = imp$sum
#      imp_data = cbind.data.frame(imp.x, imp.y)
#      imp_data = imp_data[order(imp_data$imp.y),]
#      return(imp_data)
# }

imp.rf = importance.rf$importance
imp.rf = transform(imp.rf, sum = rowSums(imp.rf))
imp.rf.x = row.names(imp.rf)
imp.rf.y = imp.rf$sum
imp_data.rf = cbind.data.frame(imp.rf.x, imp.rf.y)
imp_data.rf = imp_data.rf[order(imp_data.rf$imp.rf.y),]

#importance.rf.sum <- getImp(importance.rf)

par(mai = c(0.5,2.3,0.2,0.8))
barplot(imp_data.rf$imp.rf.y, main = "Importatnt Factors", 
        horiz = TRUE, names.arg = imp_data.rf$imp.rf.x, las=1,
        cex.axis = 1.28, cex.names = 1.28, cex.main = 1.5)


#part3: Neural network**********************************************************
control.nn <- trainControl(method="repeatedcv", number=5, repeats=3,
                           savePredictions = TRUE)

model.nn <- train(Target~., data=data.norm, method="nnet", preProcess="scale",
                  trControl=control.nn)

importance.nn <- varImp(model.nn, scale=TRUE)

imp.nn = as.data.frame(importance.nn$importance)
imp.nn.x = row.names(imp.nn)
imp.nn.y = imp.nn$Overall
imp_data.nn = cbind.data.frame(imp.nn.x, imp.nn.y)
imp_data.nn = imp_data.nn[order(imp_data.nn$imp.nn.y),]

par(mai = c(0.5,2.3,0.2,0.8))
barplot(imp_data.nn$imp.nn.y, main = "Importatnt Factors", 
        horiz = TRUE, names.arg = imp_data.nn$imp.nn.x, las=1,
        cex.axis = 1.28, cex.names = 1.28, cex.main = 1.5)


#Sum the importance of the three models*****************************************
# imp.nn$Variables <- row.names(imp.nn)
# imp.rf$Variables <- row.names(imp.rf)
# imp.svm$Variables <- row.names(imp.svm)
# 
# imp.final <- join_all(list(imp.nn, imp.rf, imp.svm),by="Variables", 
#                       type = "full")
# imp.final <- imp.final[c("Variables","Overall","sum")]

imp.overall <- imp.nn$Overall + imp.rf$sum + imp.svm$sum/6
imp.variable <- row.names(imp.nn)

imp.final <- list()
imp.final$Var <- imp.variable
imp.final$Imp <- imp.overall/3
imp.final <- as.data.frame(imp.final)

imp.final = imp.final[order(imp.final$Imp),]

par(mai = c(0.5,2.3,0.2,0.8))
barplot(imp.final$Imp, main = "Importatnt Factors", 
        horiz = TRUE, names.arg = imp.final$Var, las=1,
        cex.axis = 1, cex.names = 1, cex.main = 1)

a = predict(model.nn,newdata = data.validation)

b = data.validation$Target

c= cbind.data.frame(a,b)

#Ensemble
set.seed(500) 
rm(list = ls())

library(h2o)
library(h2oEnsemble)
h2o.init(nthreads=7, max_mem_size='20G')
h2o.removeAll()
df<-read.csv('data/ExportFile.csv')

names(df)[1]<-'ACT'

data_full<-df[,]

data_full$Month<-as.factor(data_full$Month)
######################################
YearStart<-2004
  RowStart1999<-1
  RowStart2004<-1758
  
  
  if(YearStart==1999) {RowStart<-RowStart1999}else if (YearStart==2004){RowStart<-RowStart2004}
YearEnd<-2014
RowEnd2014<-4671
RowEnd2015<-4938
if(YearEnd==2014) {RowEnd<-RowEnd2014}else if (YearEnd==2015){RowEnd<-RowEnd2015}


############## LIMIT DATA from ROWSTART TO ROWEND
data<-data_full[(RowStart:RowEnd),]

data.h2o<-as.h2o(data,destination_frame = "data.h2o")

data_full.h2o<-as.h2o(data_full, destination_frame = "data_full.h2o")

head(data)
tail(data)


predictors<-c(3:(ncol(data)))   #3:12

response <- 1


data.split<-h2o.splitFrame(data.h2o, ratios=c(0.8), destination_frames = c("train_r.h2o","valid_r.h2o") )


train_r.h2o <- data.split[[1]]

valid_r.h2o <- data.split[[2]]

#''''''''''''''''''''''''''''''''''''''

nfolds <- 22  
#'''''''''''''''' LOAD GBM '''''''''''''  
#glm_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models\\R_ADHOC_Cross_GLM_Data_Since_2004_Data_till_2014_year_end')
glm_model_r <- h2o.glm(model_id = 'GLMmid', training_frame=train_r.h2o, validation_frame=valid_r.h2o, x=predictors, y=response,max_iterations=5000,nfolds = nfolds, fold_assignment = 'Modulo', keep_cross_validation_predictions = TRUE)

gbm_model_r <- h2o.gbm(model_id = 'GBMmid', training_frame=train_r.h2o, validation_frame=valid_r.h2o, x=predictors, y=response, ntrees=500, learn_rate=0.01, score_each_iteration = TRUE,seed = 1, nfolds = nfolds, fold_assignment = 'Modulo', keep_cross_validation_predictions = TRUE)
# gbm_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models\\R_ADHOC_Cross_GBM_Data_Since_2004_Data_till_2014_year_end')

#dnn_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models\\R_ADHOC_Cross_DNN_Data_Since_2004_Data_till_2014_year_end')
dnn_model_r <- h2o.deeplearning(model_id = 'DNNmid', training_frame=train_r.h2o, validation_frame=valid_r.h2o, x=predictors, y=response, epochs=5000, hidden=c(800,800), activation =('Maxout'), l1=0, l2=0,nfolds = nfolds, fold_assignment = 'Modulo', keep_cross_validation_predictions = TRUE)

#''''''''''''''''''''''''''''''''''''''
#'''''' Ensemble Stacking '''''''''''''''''''''
 models <- list(glm_model_r, gbm_model_r, dnn_model_r)
# models <- list(gbm_model_r)


metalearner <- "h2o.gbm.wrapper"

stack <- h2o.stack(models = models, response_frame = train_r.h2o[,response], metalearner = metalearner, seed = 1, keep_levelone_data = TRUE)



perf <- h2o.ensemble_performance(stack, newdata = data_full.h2o[(5786:nrow(data_full)),])

print(perf)

#''''''''''''''''''''''''''''''''''''''
pr.glm<- h2o.predict(glm_model_r, data_full.h2o[,predictors])

pr.glm_fd_r<- (as.data.frame(pr.glm$pred))[,1]

pr.gbm<- h2o.predict(gbm_model_r, data_full.h2o[,predictors])
pr.gbm_fd_r<- (as.data.frame(pr.gbm$pred))[,1]

pr.dnn<- h2o.predict(dnn_model_r, data_full.h2o[,predictors])
pr.dnn_fd_r <- as.data.frame(pr.dnn$pred)[,1]

pr.stack<- predict(stack, data_full.h2o[,predictors])

#pr.stack<- h2o.predict(stack, data_full.h2o)


pr.stack_fd_r <- as.data.frame(pr.stack$pred)[,1]
#head(as.data.frame(gbm_model_r@model$scoring_history)) 
pr.stack_fd_r
######################
#'''''''''''''''''''' End GBM
export_df_pr_fd_r<-cbind(predict_GLM=pr.glm_fd_r,predict_GBM=pr.gbm_fd_r,predict_DNN=pr.dnn_fd_r,predict_Stack=pr.stack_fd_r, data_full)

write.csv(export_df_pr_fd_r, file='results/ExportFile_PRED_COMBINED.csv', row.names=FALSE)

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''      
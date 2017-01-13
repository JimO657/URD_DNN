set.seed(500) 
rm(list = ls())

library(h2o)
h2o.init(nthreads=7, max_mem_size='20G')
h2o.removeAll()
df<-read.csv('data/ExportFile.csv')

names(df)[1]<-'ACT'

data_full<-df[,]

data_full$Month<-as.factor(data_full$Month)

YearStart<-2004

YearEnd<-2014

data_full.h2o<-as.h2o(data_full, destination_frame = 'data_full.h2o')

predictors<-c(3:(ncol(data_full)))
predictors
response <- 1
response

#'''''''''''''''' LOAD GBM '''''''''''''  
glm_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models\\R_ADHOC_GLM_Data_Since_2004_Data_till_2014_year_end')

gbm_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models\\R_ADHOC_GBM_Data_Since_2004_Data_till_2014_year_end')
  
dnn_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models\\R_ADHOC_DNN_Data_Since_2004_Data_till_2014_year_end')


pr.glm<- h2o.predict(glm_model_r, data_full.h2o[,predictors])

pr.glm_fd_r<- (as.data.frame(pr.glm$pred))[,1]

pr.gbm<- h2o.predict(gbm_model_r, data_full.h2o[,predictors])
pr.gbm_fd_r<- (as.data.frame(pr.gbm$pred))[,1]

pr.dnn<- h2o.predict(dnn_model_r, data_full.h2o[,predictors])
pr.dnn_fd_r <- as.data.frame(pr.dnn$pred)[,1]
 
#head(as.data.frame(gbm_model_r@model$scoring_history)) 

######################
#'''''''''''''''''''' End GBM
export_df_pr_fd_r<-cbind(predict_GLM=pr.gbm_fd_r,predict_GBM=pr.gbm_fd_r,predict_DNN=pr.dnn_fd_r, data_full)

write.csv(export_df_pr_fd_r, file='results/ExportFile_PRED_COMBINED.csv', row.names=FALSE)

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''            
library(h2o)
h2o.init(nthreads=-1, max_mem_size='15G')
h2o.removeAll()

df<-read.csv('data/ExportFileWeather_2007.csv')
df2<-read.csv('data/ExportFile.csv')


names(df)[1]<-'ACT'
names(df2)[1]<-'ACT'

data_full<-df[,]

data_full$Month1<-as.factor(data_full$Month1)

data_full2<-df2[,]

data_full2$Month1<-as.factor(data_full2$Month1)
        
YearStart<-2006 #"&mrvStartYear #&vStartYear '
###################################### 
YearEnd<-2014

data_full.h2o<-as.h2o(data_full, destination_frame = 'data_full.h2o')
data_full2.h2o<-as.h2o(data_full2, destination_frame = 'data_full2.h2o')

#WriteLog("Weather Predict data_full Loaded "&Runtime&".")
predictors<-c(3:(ncol(data_full)))
predictors
response <- 1
response

######## LOAD Models ####### 
glm_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models\\R_ADHOC_GLM_Data_Since_2006_Data_till_2014_year_end')
#WriteLog("Weather GLM Model Loaded "&Runtime&". "&Runtime&".")
#           gbm_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models\\R_ADHOC_GBM_Data_Since_"&GBMStartYear&"_Data_till_2014_year_end')
#    #WriteLog("Weather GBM Model Loaded "&Runtime&".")  
#           dnn_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models\\R_ADHOC_DNN_Data_Since_"&DNNStartYear&"_Data_till_2014_year_end')
#    #WriteLog("Weather DNN Model Loaded "&Runtime&".")
#           best_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models\\R_ADHOC_Rand_DNN_Data_Since_2006_Data_till_2014_year_end')
#    #WriteLog("Weather BEST Model Loaded "&Runtime&".")

pr.glm<- h2o.predict(glm_model_r, data_full.h2o[,predictors])
pr.glm_fd_r<- (as.data.frame(pr.glm$pred))[,1]

pr.glm2<- h2o.predict(glm_model_r, data_full2.h2o[,predictors])
pr.glm2_fd_r<- (as.data.frame(pr.glm$pred))[,1]

#    #WriteLog("GLM Model Predicted "&Runtime&".")
#           pr.gbm<- h2o.predict(gbm_model_r, data_full.h2o[,predictors])
#           pr.gbm_fd_r<- (as.data.frame(pr.gbm$pred))[,1]


#            pr.dnn<- h2o.predict(dnn_model_r, data_full.h2o[,predictors])
#            pr.dnn_fd_r<-as.data.frame(pr.dnn$pred)[,1]
#           
#            pr.best<- h2o.predict(best_model_r, data_full.h2o[,predictors])
#            pr.best_fd_r<-as.data.frame(pr.best$pred)[,1]

########## Write out 
export_df_pr_fd_r<-cbind(predict_WEATHER=pr.glm_fd_r,predict_GLM=pr.glm2_fd_r, data_full)
min_temp<-cbind(data_full$Min_Temp,data_full2$Min_Temp)


write.csv(export_df_pr_fd_r, file='results/ExportFile_PRED_WEATHER.csv', row.names=FALSE)
#WriteLog("Weather Predictions written "&Runtime&".")
################################'
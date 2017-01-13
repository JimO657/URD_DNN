set.seed(500)     
rm(list = ls())
#WriteLog("Predict Macro Started "&Runtime&".")     
# Set myTable=ActiveDocument.GetSheetObject("DataSentToR")
# myTable.Export CurrentPath&"/data_2015/ExportFile.csv",",",0     
#WriteLog("Predict Table exported "&Runtime&".")
#' Create a COM object representing R
# Set R = CreateObject("StatConnectorSrv.StatConnector")
# R.Init "R"
#' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' '
#WriteLog("Predict R Started "&Runtime&".")
library(h2o)
h2o.init(nthreads=7, max_mem_size='40G')
h2o.removeAll()
#WriteLog("Predict H2O Started "&Runtime&".") 
df<-read.csv('data_2015/ExportFileR.csv')

names(df)[1]<-'ACT'

data_full<-df[,]

data_full$Month1<-as.factor(data_full$Month1)


#' #' #' #' #' #' '#' Setting variables  #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' 

#WriteLog("Predict about to set variables "&Runtime&". ")
mrvStartYear = 2006
mrvEndYear =   2014
GLMStartYear = 2006
GBMStartYear = 2006
DNNStartYear = 2006
GLMEndYear = 2014
GBMEndYear = 2014
DNNEndYear = 2014
#' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' 
#WriteLog("Predict about to set YearStart to mrvStartYear "&Runtime&". ")			            
YearStart<-mrvStartYear #' &vStartYear '
#' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #'  
YearEnd<-mrvEndYear
#' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #'  
data_full.h2o<-as.h2o(data_full, destination_frame = 'data_full.h2o')
#WriteLog("Predict data_full Loaded "&Runtime&".")
predictors<-c(3:(ncol(data_full)))
#WriteLog("Predict predictors created "&Runtime&".")
#'            predictors
response <- 1
#'            response

#' #' #' #' #' #' #' '#' LOAD Models #' #' #' #' #' #' #'  
#WriteLog("Predict about to load GLM model "&Runtime&".")
glm_model_r <-  h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models_2015\\R_ADHOC_GLM_Data_Since_2006_Data_till_2014_year_end')
#WriteLog("GLM Model Loaded "&Runtime&". "&Runtime&".")       
gbm_model_r <-  h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models_2015\\R_ADHOC_GBM_Data_Since_2006_Data_till_2014_year_end')
#WriteLog("GBM Model Loaded "&Runtime&".")  
dnn_model_r <-  h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models_2015\\R_ADHOC_DNN_Data_Since_2006_Data_till_2014_year_end')
#WriteLog("DNN Model Loaded "&Runtime&".")
best_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models_2015\\R_ADHOC_DNN_Data_Since_2006_Data_till_2014_year_end')
#WriteLog("BEST Model Loaded "&Runtime&".")

pr.glm<- h2o.predict(glm_model_r, data_full.h2o[,predictors])
pr.glm_fd_r<- (as.data.frame(pr.glm$pred))[,1]
#WriteLog("GLM Model Predicted "&Runtime&".")
pr.gbm<- h2o.predict(gbm_model_r, data_full.h2o[,predictors])
pr.gbm_fd_r<- (as.data.frame(pr.gbm$pred))[,1]
#WriteLog("GBM Model Predicted "&Runtime&".")

 pr.dnn<- h2o.predict(dnn_model_r, data_full.h2o[,predictors])
 pr.dnn_fd_r<-as.data.frame(pr.dnn$pred)[,1]
#WriteLog("DNN Model Predicted "&Runtime&".")            
 pr.best<- h2o.predict(best_model_r, data_full.h2o[,predictors])
 pr.best_fd_r<-as.data.frame(pr.best$pred)[,1]
#WriteLog("BEST Model Predicted "&Runtime&".")
#' #' #' #' #' #' #' #' #' '#' Write out 
export_df_pr_fd_r<-cbind(predict_BEST=pr.best_fd_r,predict_GLM=pr.glm_fd_r,predict_GBM=pr.gbm_fd_r,predict_DNN=pr.dnn_fd_r, data_full)
#WriteLog("Prediction Export frame created "&Runtime&".")
write.csv(export_df_pr_fd_r, file='results_2015/ExportFile_PRED_COMBINED.csv', row.names=FALSE)
#WriteLog("Predictions written "&Runtime&".")
#' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #' '
#'            h2o.shutdown(prompt = FALSE)
#' Close R connection

#' #' #' #' #' #' #' #' #' #' #' #' #' #' #' #'     

#' Load the scores into the QlikView application
#ActiveDocument.DoReload 2, false,false		'Fail on error, Partial Reload
#'     ActiveDocument.ActivateSheet "Results"
#WriteLog("Predict Macro Exited Succesfully "&Runtime&".")
#MsgBox("Predicted Succesfully")


#'''''''' PREDICT WEATHER YEAR GLM GBM DNN BEST''''''''''''''''''''''''
#Sub runRPredictWeahter
#WriteLog("Weather Predict Macro Started "&Runtime&".") 
#''''' Prepare the Weather data set and Ecport Variables to Text file '''''''
#Clr
#Reload
#WriteWYear
#mrwYear = ActiveDocument.Evaluate("text($(wYear))")
#''''''''''''''
#''' Export vYear set to wYear
#Set v = ActiveDocument.GetVariable("vYear")
#v.SetContent mrwYear, True

#Set myTable=ActiveDocument.GetSheetObject("DataSentToRW")
#myTable.Export CurrentPath&"/data_2015/ExportFileWeather_"&mrwYear&".csv",",",0     

#' Create a COM object representing R
#Set R = CreateObject("StatConnectorSrv.StatConnector")
#R.Init "R"
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#WriteLog("Weather Predict R Started "&Runtime&".")
library(h2o)
h2o.init(nthreads=3, max_mem_size='2G')
h2o.removeAll()
#WriteLog("Weather Predict H2O Started "&Runtime&".") 
df<-read.csv('data_2015/ExportFileWeather_2010.csv')

names(df)[1]<-'ACT'

data_full<-df[,]

data_full$Month1<-as.factor(data_full$Month1)

#'''''''''''''' Setting variables  ''''''''''''''''''''''''''''''''''''''' 

#WriteLog("Predict about to set variables "&Runtime&". ")
mrvStartYear = 2006
mrvEndYear =   2014				
GLMStartYear = 2006
GBMStartYear = 2006
DNNStartYear = 2006
GLMEndYear = 2014
GBMEndYear = 2014
DNNEndYear = 2014
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#WriteLog("Predict about to set YearStart to mrvStartYear "&Runtime&". ")			            
YearStart<-"&mrvStartYear ' &vStartYear 
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''  
YearEnd<- &mrvEndYear
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

data_full.h2o<-as.h2o(data_full, destination_frame = 'data_full.h2o')
#WriteLog("#Weather Predict data_full Loaded "&Runtime&".")
predictors<-c(3:(ncol(data_full)))
predictors
response <- 1
response

#'''''''''''''''' LOAD Models '''''''''''''  
glm_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models_2015\\R_ADHOC_GLM_Data_Since_"&GLMStartYear&"_Data_till_"&GLMEndYear&"_year_end')
#WriteLog("#Weather GLM Model Loaded "&Runtime&". "&Runtime&".")
gbm_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models_2015\\R_ADHOC_GBM_Data_Since_"&GBMStartYear&"_Data_till_"&GBMEndYear&"_year_end')
#WriteLog("#Weather GBM Model Loaded "&Runtime&".")  
#WriteLog("#Weather about to load DNN Model "&Runtime&".")
dnn_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models_2015\\R_ADHOC_DNN_Data_Since_"&DNNStartYear&"_Data_till_"&DNNEndYear&"_year_end')
#WriteLog("#Weather DNN Model Loaded "&Runtime&".")
best_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models_2015\\R_ADHOC_DNN_Data_Since_"&DNNStartYear&"_Data_till_"&DNNEndYear&"_year_end')
#WriteLog("#Weather BEST Model Loaded "&Runtime&".")

pr.glm<- h2o.predict(glm_model_r, data_full.h2o[,predictors])
pr.glm_fd_r<- (as.data.frame(pr.glm$pred))[,1]
#WriteLog("#GLM Model Predicted "&Runtime&".")
pr.gbm<- h2o.predict(gbm_model_r, data_full.h2o[,predictors])
pr.gbm_fd_r<- (as.data.frame(pr.gbm$pred))[,1]


pr.dnn<- h2o.predict(dnn_model_r, data_full.h2o[,predictors])
pr.dnn_fd_r<-as.data.frame(pr.dnn$pred)[,1]
#'            
pr.best<- h2o.predict(best_model_r, data_full.h2o[,predictors])
pr.best_fd_r<-as.data.frame(pr.best$pred)[,1]

#'''''''''''''''''''' Write out 
export_df_pr_fd_r<-cbind(predict_WBEST=pr.best_fd_r,predict_WGLM=pr.glm_fd_r,predict_WGBM=pr.gbm_fd_r,predict_WDNN=pr.dnn_fd_r, data_full)

write.csv(export_df_pr_fd_r, file='"&CurrentPath&"/results_2015/ExportFile_PRED_WEATHER.csv', row.names=FALSE)
#WriteLog("#Weather Predictions written "&Runtime&".")
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#'            R.EvaluateNoReturn "h2o.shutdown(prompt = FALSE)"
#' Close R connection
#R.close
#'''''''''''''''''''''''''''''''     

#' Load the scores into the QlikView application
#ActiveDocument.DoReload 2, false,false		'Fail on error, Partial Reload
#'     ActiveDocument.ActivateSheet "Results"
#WriteLog("Predict Macro Exited Succesfully "&Runtime&".")
#MsgBox("Weather Year "& mrwYear& " Predicted Succesfully")

#End Sub
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

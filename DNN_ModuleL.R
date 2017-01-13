set.seed(500)     
rm(list = ls())
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#Sub runRDNN
#WriteLog("DNN Training macro started at "&Runtime&". ")
#' Export the selected patient data to be scored.
#Set myTable=ActiveDocument.GetSheetObject("DataSentToR")
#myTable.Export CurrentPath&"/data_2015/ExportFile.csv",",",0     

#' Create a COM object representing R
#Set R = CreateObject("StatConnectorSrv.StatConnector")
#R.Init "R"
#''''''''''''''''''''''''''''''''''	
library(h2o)
h2o.init(nthreads=7, max_mem_size='40G')
#WriteLog("DNN H2O started at "&Runtime&". ")
h2o.removeAll()
df<-read.csv('data_2015/ExportFileR.csv')

#WriteLog("DNN datat read at "&Runtime&". ")           
names(df)[1]<-'ACT'

data_full<-df[,]
data_full$Month1<-as.factor(data_full$Month1)


#'''''''''''''' Setting variables 

#WriteLog("DNN about to set variables "&Runtime&". ")
#'            mrv2014 = ActiveDocument.Evaluate("text($(vRowYearEnd2014))")
#'			mrv2015 = ActiveDocument.Evaluate("text($(vRowYearEnd2015))")
mrvStartYear = 2006
mrvEndYear =2014
#Set v = 2006
v = 2006
#v.SetContent mrvStartYear, True
#Set ve = 2014
ve = 2014
#ve.SetContent mrvEndYear, True	 				 			
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''              
#'          #### Choose Start end end years to use for training 
#WriteLog("DNN about to set YearStart to mrvStartYear "&Runtime&". ")			            
YearStart<-mrvStartYear 

#'           R.EvaluateNoReturn "YearStart<-2006"
#WriteLog("DNN about to set YearEnd to mrvEndYear "&Runtime&". ")			            
YearEnd<-mrvEndYear 
#'           R.EvaluateNoReturn "YearEnd<-2015"


ds<- paste0(YearStart,'-Jan-01')

RowStart<-which(data_full$Date1 == ds)
#'RowStart

de<- paste0(YearEnd,'-Dec-31')

RowEnd<-which(data_full$Date1 == de)
#'RowEnd
#WriteLog("DNN row numbers set "&Runtime&". ")			            
#'##################################################
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''' 

data<-data_full[(RowStart:RowEnd),]
data.h2o<-as.h2o(data,destination_frame = 'data.h2o')
data_full.h2o<-as.h2o(data_full, destination_frame = 'data_full.h2o')
predictors<-c(3:(ncol(data)))
predictors
response <- 1
response
data.split<-h2o.splitFrame(data.h2o, ratios=c(0.8), destination_frames = c('train_r.h2o','valid_r.h2o') )
train_r.h2o <- data.split[[1]]
str(train_r.h2o)

valid_r.h2o <- data.split[[2]]
#'''''''''''''''' Start DNN '''''''''''''  
mid<-paste('R_ADHOC','DNN','Data_Since',YearStart,'Data_till',YearEnd,'year_end',sep = '_')
#WriteLog("DNN Training started at "&Runtime&". ")
#dnn_model_r <- h2o.deeplearning(model_id = mid, training_frame=train_r.h2o, validation_frame=valid_r.h2o, x=predictors, y=response, epochs=5000, hidden=c(800,800), activation =('Maxout'), l1=0, l2=0,stopping_metric= 'MSE',stopping_rounds=5,stopping_tolerance=1e-6)
#h2o.deeplearning(model_id = mid, training_frame=train_r.h2o, validation_frame=valid_r.h2o, x=predictors, y=response, epochs=5000, hidden=c(800,800), activation =('Maxout'), l1=0, l2=0,stopping_metric= 'MSE',stopping_rounds=25,stopping_tolerance=1e-6)
### REmoving early stopping
h2o.deeplearning(model_id = mid, training_frame=train_r.h2o, validation_frame=valid_r.h2o, x=predictors, y=response, epochs=500, hidden=c(800,800), activation =('Tanh'), l1=0, l2=0,stopping_rounds=0,stopping_metric= 'MSE',stopping_tolerance=1e-6)

#'		    R.EvaluateNoReturn "dnn_model_r <- h2o.deeplearning(model_id = mid, training_frame=train_r.h2o, validation_frame=valid_r.h2o, x=predictors, y=response, epochs=5000, hidden=c(5000),    activation =('MaxoutWithDropout'), l1=(1e-6),l2=(1e-6),stopping_metric= 'MSE',stopping_rounds=20,stopping_tolerance=1e-6)"
#''''''''''''''''' Enable Cross Validation ''''''''''''''''     
#'            R.EvaluateNoReturn "nfolds <- 5" 
#'			R.EvaluateNoReturn "dnn_model_r <- h2o.deeplearning(model_id = mid, training_frame=train_r.h2o, validation_frame=valid_r.h2o, x=predictors, y=response, epochs=5000, hidden=c(800,800), activation =('Maxout'), l1=0, l2=0,nfolds = nfolds, fold_assignment = 'Modulo', keep_cross_validation_predictions = TRUE)"

#WriteLog("DNN Training finished at "&Runtime&". ")
#'''''''''''''''''''' End DNN

#''''''''''' Save Model and POJO ''''''''''''''''
savedDNN_Rand<-h2o.saveModel(dnn_model_r, path = '/0MyDataBases/7R/ADHOC_Qlikview/H2O_Models_2015', force = TRUE)      
savedDNN_Rand
#WriteLog("DNN model saved at "&Runtime&". ")
savedDNN_Rand_POJO<-h2o.download_pojo(dnn_model_r, path = '/0MyDataBases/7R/ADHOC_Qlikview/POJO_2015', getjar = TRUE)
#WriteLog("DNN POJO saved at "&Runtime&". ")
#'''''''''''''''''''''''''''''''

#''''''''''''''' Setting additional variables  ''''''''''''''''''''''''''''''''''''''' 
#'			GLMStartYear = ActiveDocument.Evaluate("text($(vMidGLM))")
#'			GBMStartYear = ActiveDocument.Evaluate("text($(vMidGBM))")
#'			DNNStartYear = ActiveDocument.Evaluate("text($(vMidDNN))")
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''  
#'   '''''''''''''''' LOAD Models '''''''''''''
#'     
#'            R.EvaluateNoReturn "glm_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models_2015\\R_ADHOC_GLM_Data_Since_"&GLMStartYear&"_Data_till_2014_year_end')"
#'     WriteLog("GLM Model Loaded "&Runtime&". "&Runtime&".")
#'     
#'            R.EvaluateNoReturn "gbm_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models_2015\\R_ADHOC_GBM_Data_Since_"&GBMStartYear&"_Data_till_2014_year_end')"
#'     WriteLog("GBM Model Loaded "&Runtime&".")  
#'
#''            R.EvaluateNoReturn "dnn_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models_2015\\R_ADHOC_DNN_Data_Since_"&DNNStartYear&"_Data_till_2014_year_end')"
#''     WriteLog("DNN Model Loaded "&Runtime&".")
#'     
#'            R.EvaluateNoReturn "best_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models_2015\\0Best_R_ADHOC_GBM_Data_Since_2006_Data_till_2014_year_end')"
#'     WriteLog("BEST Model Loaded "&Runtime&".")
#'   ''''''''''''''''''''''''''''''''''''''''''''''  
#'   ''''''''''' Create Predictions form loaded models ''''''''''''''''''        
#'            R.EvaluateNoReturn "pr.glm<- h2o.predict(glm_model_r, data_full.h2o[,predictors])"
#'            R.EvaluateNoReturn "pr.glm_fd_r<- (as.data.frame(pr.glm$pred))[,1]"
#'     WriteLog("GLM Model Predicted "&Runtime&".")
#'     
#'            R.EvaluateNoReturn "pr.gbm<- h2o.predict(gbm_model_r, data_full.h2o[,predictors])"
#'            R.EvaluateNoReturn "pr.gbm_fd_r<- (as.data.frame(pr.gbm$pred))[,1]"
#'     WriteLog("GBM Model Predicted "&Runtime&".")
#'     
#'            R.EvaluateNoReturn " pr.dnn<- h2o.predict(dnn_model_r , data_full.h2o[,predictors])"
#'            R.EvaluateNoReturn " pr.dnn_fd_r<-as.data.frame(pr.dnn$pred)[,1]"
#'     WriteLog("DNN Model predicted at "&Runtime&". ")  
#'               
#'            R.EvaluateNoReturn " pr.best<- h2o.predict(best_model_r, data_full.h2o[,predictors])"
#'            R.EvaluateNoReturn " pr.best_fd_r<-as.data.frame(pr.best$pred)[,1]"
#'     WriteLog("DNN best model predicted at "&Runtime&". ")
#'     
#'
#'     
#''''''''''''''''''''' Write out 
#'            R.EvaluateNoReturn "export_df_pr_fd_r<-cbind(predict_BEST=pr.best_fd_r,predict_GLM=pr.glm_fd_r,predict_GBM=pr.gbm_fd_r,predict_DNN=pr.dnn_fd_r, data_full)"
#'
#'            R.EvaluateNoReturn "write.csv(export_df_pr_fd_r, file='"&CurrentPath&"/results_2015/ExportFile_PRED_COMBINED.csv', row.names=FALSE)"
#'     WriteLog("DNN CSV written at "&Runtime&". ")   
#'   
#''''''''''''''''''''''''''' Plots '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''            
setwd('/0MyDataBases/7R/ADHOC_Qlikview/plots_2015')
png('Score_Plot_DNN.png')
plot(dnn_model_r,metric='MSE')
title(main='DNN',  col.main='black', line='3', font.main=4)
dev.off()
setwd('/0MyDataBases/7R/ADHOC_Qlikview')
#WriteLog("DNN plot saved at "&Runtime&". ")
#'            R.EvaluateNoReturn "setwd('..') "           
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''    
#'           R.EvaluateNoReturn "h2o.shutdown(prompt = FALSE)"
#' Close R connection
#R.close
#'''''''''''''''''''''''''''''''     
#Set v = ActiveDocument.GetVariable("vRuntime")
#v.SetContent Runtime(), True

#' Load the scores into the QlikView application
#'     ActiveDocument.DoReload 2, false,false		'Fail on error, Partial Reload
#'     ActiveDocument.ActivateSheet "Results"
#WriteLog("DNN Training completed at "&Runtime&". ")
#'     MsgBox("DNN Trained Succesfully")	
#End Sub
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dnn_model_r
plot(dnn_model_r,metric='MSE')
time <- dnn_model_r@model$run_time/1000/60
print(paste0('DNN Time taken ', time))

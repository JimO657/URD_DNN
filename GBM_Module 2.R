 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #'
     #   #  WriteLog("Predict R Started "&Runtime&".")
 library(h2o)
 h2o.init(nthreads=-1, max_mem_size='15G')
 h2o.removeAll()
 #  WriteLog("Predict H2O Started "&Runtime&".") 
 df<-read.csv('data/ExportFile.csv')

 names(df)[1]<-'ACT'

 data_full<-df[,]

 data_full$Month1<-as.factor(data_full$Month1)

 #            YearStart<-2004
 # # # # # #' #Setting variables   # # # # # # # # # # # # # # # # # # # #

 #  WriteLog("GLM about to set variables "&Runtime&". ")
# mrv2014 = ActiveDocument.Evaluate("text($(vRowYearEnd2014))")
# mrv2015 = ActiveDocument.Evaluate("text($(vRowYearEnd2015))")
# mrvStartYear = ActiveDocument.Evaluate("text($(vStartYear))")
# 
# GLMStartYear = ActiveDocument.Evaluate("text($(vMidGLM))")
# GBMStartYear = ActiveDocument.Evaluate("text($(vMidGBM))")
# DNNStartYear = ActiveDocument.Evaluate("text($(vMidDNN))")
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 #  WriteLog("GLM about to set YearStart to mrvStartYear "&Runtime&". ")			            
 YearStart<- 2006 
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
 YearEnd<-2014

 data_full.h2o<-as.h2o(data_full, destination_frame = 'data_full.h2o')
 #  WriteLog("Predict data_full Loaded "&Runtime&".")
 
 
 nrow(data_full)
 nrow(data_full.h2o) 
 
 
 predictors<-c(3:(ncol(data_full)))
 predictors
 response <- 1
 response

 # # # # # # #' #LOAD Models  # # # # # # # 
 glm_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models\\R_ADHOC_GLM_Data_Since_2006_Data_till_2014_year_end')
 #  WriteLog("GLM Model Loaded "&Runtime&". "&Runtime&".")
 gbm_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models\\R_ADHOC_GBM_Data_Since_2006_Data_till_2014_year_end')
 #  WriteLog("GBM Model Loaded "&Runtime&".")  
 dnn_model_r <-  h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models\\R_ADHOC_DNN_Data_Since_2006_Data_till_2014_year_end')
 #  WriteLog("DNN Model Loaded "&Runtime&".")
 best_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models\\R_ADHOC_GBM_Data_Since_2006_Data_till_2014_year_end')
 #  WriteLog("BEST Model Loaded "&Runtime&".")

 pr.glm<- h2o.predict(glm_model_r, data_full.h2o[,predictors])
 pr.glm_fd_r<- (as.data.frame(pr.glm$pred))[,1]
 #  WriteLog("GLM Model Predicted "&Runtime&".")
 pr.gbm<- h2o.predict(gbm_model_r, data_full.h2o[,predictors])
 pr.gbm_fd_r<- (as.data.frame(pr.gbm$pred))[,1]
 #  WriteLog("GBM Model Predicted "&Runtime&".")

  pr.dnn<- h2o.predict(dnn_model_r, data_full.h2o[,predictors])
  pr.dnn_fd_r<-as.data.frame(pr.dnn$pred)[,1]
 #  WriteLog("DNN Model Predicted "&Runtime&".")            
  pr.best<- h2o.predict(best_model_r, data_full.h2o[,predictors])
  pr.best_fd_r<-as.data.frame(pr.best$pred)[,1]
 #  WriteLog("BEST Model Predicted "&Runtime&".")
 # # # # # # # # #' #Write out 

  
 export_df_pr_fd_r<-cbind(predict_BEST=pr.best_fd_r,predict_GLM=pr.glm_fd_r,predict_GBM=pr.gbm_fd_r,predict_DNN=pr.dnn_fd_r, data_full)
 #  WriteLog("Prediction Export frame created "&Runtime&".")
 write.csv(export_df_pr_fd_r, file='"&CurrentPath&"/results/ExportFile_PRED_COMBINED.csv', row.names=FALSE)
 #  WriteLog("Predictions written "&Runtime&".")
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #'
 #            h2o.shutdown(prompt = FALSE)"
 #Close R connection

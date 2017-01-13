 library(h2o)
 h2o.init(nthreads=7, max_mem_size='20G')
 h2o.removeAll()
 df<-read.csv('data/ExportFile.csv')

 names(df)[1]<-'ACT'

 data_full<-df[,]

 data_full$Month<-as.factor(data_full$Month)

 #### Choose Start end end years to use for training 
 YearStart<-2006
 
 YearEnd<-2014
 
 
 ds<- paste0(YearStart,'-Jan-01')
 
 RowStart<-which(data_full$Date1 == ds)
 RowStart
 
 de<- paste0(YearEnd,'-Dec-31')
 
 RowEnd<-which(data_full$Date1 == de)
 RowEnd
 ########################################

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
#'''''''''''''''' Start GBM '''''''''''''  
 mid<-paste('R_ADHOC','Rand','GBM','Data_Since',YearStart,'Data_till',YearEnd,'year_end',sep = '_')

#' best_model_r <- h2o.deeplearning(model_id = mid, training_frame=train_r.h2o, validation_frame=valid_r.h2o, x=predictors, y=response, epochs=5000, stopping_metric='MSE', hidden=c(800,800), activation =('Maxout'))"
 gbm_model_r <- h2o.gbm(model_id = mid, training_frame=train_r.h2o, validation_frame=valid_r.h2o, x=predictors, y=response, ntrees=500, learn_rate=0.01, score_each_iteration = TRUE)
 pr.h2o_gbm_fd_r<- h2o.predict(gbm_model_r, data_full.h2o[,predictors])
 pr.gbm_fd.df_r<-  as.data.frame(pr.h2o_gbm_fd_r)
 pr.gbm_fd_r<-pr.gbm_fd.df_r[[1]]

 head(as.data.frame(gbm_model_r@model$scoring_history)) 
 ######## Plot
 setwd('/0MyDataBases/7R/ADHOC_Qlikview/plots') 
 png('Score_Plot_GBM.png')
 plot(gbm_model_r,metric='MSE')
 title(main='GBM',  col.main='black', line='3', font.main=4)
# devOptions("*", force=TRUE)
 dev.off()
 setwd('/0MyDataBases/7R/ADHOC_Qlikview')
 ######################
 
savedGBM_Rand<-h2o.saveModel(gbm_model_r, path = '/0MyDataBases/7R/ADHOC_Qlikview/H2O_Models', force = TRUE)            
 savedGBM_Rand
savedGBM_Rand_POJO<-h2o.download_pojo(gbm_model_r, path = '/0MyDataBases/7R/ADHOC_Qlikview/POJO', getjar = TRUE)
#'''''''''''''''''''' End GBM
 export_df_pr_fd_r<-cbind(predict_GBM=pr.gbm_fd_r, data_full)

 write.csv(export_df_pr_fd_r, file='results/ExportFile_PRED_GBM.csv', row.names=FALSE)
#'''''''''''''''''''''''''' Plots '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''            
           
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''            
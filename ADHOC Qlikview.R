#Saved on Workstation
set.seed(500) 
rm(list = ls())
df<-read.csv("data/ExportFile.csv")


library(h2o)
h2o.init(nthreads=7, max_mem_size="50G")
h2o.removeAll() ## clean slate - just in case the cluster was already running



############################################################################
######## Breaking Data into train, validate and test
############################################################################

names(df)[1]<-"ACT"

data_full<-df[,]

#data$RPT_YR<-as.factor(data$RPT_YR)  
data_full$Month1<-as.factor(data_full$Month1) 



###############################################################
#### Choose Start end end years to use for training 
YearStart<-2006

YearEnd<-2014


ds<- paste0(YearStart,'-Jan-01')

RowStart<-which(data_full$Date1 == ds)
RowStart

de<- paste0(YearEnd,'-Dec-31')

RowEnd<-which(data_full$Date1 == de)
RowEnd
########################################################    
   
    
    ############## LIMIT DATA from ROWSTART TO ROWEND
    data<-data_full[(RowStart:RowEnd),]
    
    data.h2o<-as.h2o(data,destination_frame = "data.h2o")
    
    data_full.h2o<-as.h2o(data_full, destination_frame = "data_full.h2o")
    
#     head(data)
#     tail(data)
    
    
    predictors<-c(3:(ncol(data)))   #3:12
    # predictors
    
     # str(data)
     str(data[,predictors])
    response <- 1
    # response
    
    

    ####################################################################################
    ####################################################################################
    ####################################################################################
    ####################################################################################
    #############################################################################
    ###### Random splits 
    ############################################################################
    
    data.split<-h2o.splitFrame(data.h2o, ratios=c(0.8), destination_frames = c("train_r.h2o","valid_r.h2o") )
    #head(data.split)
    #tail(data.split)
    
    train_r.h2o <- data.split[[1]]
    # str(train_r.h2o)
    
#     head(train_r.h2o)
#     tail(train_r.h2o)
    
    valid_r.h2o <- data.split[[2]]
#     head(valid_r.h2o)
#     tail(valid_r.h2o)
    
    
    #test_r.h2o <- data.split[[3]]
    #str(test_r.h2o)
    
    RS<-RowEnd+1
    RE<-nrow(data_full.h2o)
    
    test<-data_full[(RS:RE),]
    
    test.h2o<-as.h2o(test,destination_frame = 'test.h2o')
#     head(test.h2o)
#     tail(test.h2o)

    #######################################################################################
    ############ RANDOM SINGLE DNN 
    ############################################################################
    
    mid<-paste('R_ADHOC','Rand','DNN','Data_Since',YearStart,'Data_till',YearEnd,'year_end',sep = '_')
    mid
    start.time <- Sys.time()
    
    dnn_model_r1 <- h2o.deeplearning(
      model_id = mid,
      training_frame=train_r.h2o,
      validation_frame=valid_r.h2o, 
      x=predictors, 
      y=response,
      epochs=5000,
      stopping_metric="MSE",
      hidden=c(5000),   #c(8000,1000),
        l1=(1e-6),
        l2=(1e-6),
      activation =("MaxoutWithDropout")
        
    )
    
    end.time <- Sys.time()
    time.taken <- end.time - start.time
    time.taken
    
   # savedDNN_Rand_R<-h2o.saveModel(dnn_model_r1, path = "H2O_Models", force = TRUE)
   #  savedDNN_Rand_R_POJO<-h2o.download_pojo(dnn_model_r1, path = "POJO", getjar = TRUE)
    
    time <- dnn_model_r1@model$run_time/1000/60
    print(paste0("training time in minutes: ", time))
    
    ##############################################################
    ###### Check the performance of the models ##############
    ## Check performance of the first model
    h2o.performance(model = dnn_model_r1, newdata = test.h2o)
    
    

    dnn_model_r1@model$training_metrics@metrics$MSE
    dnn_model_r1@model$validation_metrics@metrics$MSE
    

    
    
    
    ###############################################################
    ############# Predicting with best model
    ############################################################################
    pr.dnn_best<- h2o.predict(dnn_model_r1, data_full.h2o[,predictors])
    pr.dnn_best_fd_r <- as.data.frame(pr.dnn_best$pred)[,1]
    
#     pr.h2o_dnn_fd_r<- h2o.predict(dnn_model_r1, data_full.h2o[,predictors])
#     pr.dnn_fd.df_r<-  as.data.frame(pr.h2o_dnn_fd_r)
#     pr.dnn_fd_r1<-pr.dnn_fd.df_r[[1]]
    
    ####################################################################################################
    ############# parameters and save the model
    ####################################################################################################
    best_params_r <- dnn_model_r1@allparameters
    Best_model_param_desc_fd_r<-paste0('R ADHOC ','Rand ','Data Starts at ',YearStart,' Data Ends at ',YearEnd,' ',best_params_r$activation,' ',toString(best_params_r$hidden),' ','l1=',best_params_r$l1,' l2=',best_params_r$l2, ' training time ', round(time,2), ' minutes',sep="")
    Best_model_param_desc_fd_r
    
    savedDNN_b<-h2o.saveModel(dnn_model_r1, path = '/0MyDataBases/7R/ADHOC_Qlikview/H2O_Models', force = TRUE)            
    savedDNN_b
   
    ############################################################
    glm_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models\\R_ADHOC_GLM_Data_Since_2006_Data_till_2014_year_end')
    
    gbm_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models\\R_ADHOC_GBM_Data_Since_2006_Data_till_2014_year_end')
    
    dnn_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models\\R_ADHOC_DNN_Data_Since_2006_Data_till_2014_year_end')
    
    best_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models\\R_ADHOC_Rand_DNN_Data_Since_2006_Data_till_2014_year_end')
    
    pr.glm<- h2o.predict(glm_model_r, data_full.h2o[,predictors])
    pr.glm_fd_r<- (as.data.frame(pr.glm$pred))[,1]
    
    pr.gbm<- h2o.predict(gbm_model_r, data_full.h2o[,predictors])
    pr.gbm_fd_r<- (as.data.frame(pr.gbm$pred))[,1]
    
    pr.dnn<- h2o.predict(dnn_model_r, data_full.h2o[,predictors])
    pr.dnn_fd_r <- as.data.frame(pr.dnn$pred)[,1]
 
    pr.best<- h2o.predict(best_model_r, data_full.h2o[,predictors])
    pr.best_fd_r<-as.data.frame(pr.best$pred)[,1]
    
    #head(as.data.frame(gbm_model_r@model$scoring_history)) 
    
    ######################
    #'''''''''''''''''''' End GBM
    export_df_pr_fd_r<-cbind(predict_BEST=pr.dnn_best_fd_r,
                             predict_DNN=pr.dnn_fd_r,
                             predict_GLM=pr.glm_fd_r,
                             predict_GBM=pr.gbm_fd_r,
                             data_full)
    
    write.csv(export_df_pr_fd_r, file='results/ExportFile_PRED_COMBINED.csv', row.names=FALSE)
    
    # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''     
    
    print(paste('Exported to results/ExportFile_PRED_COMBINED.csv',Best_model_param_desc_fd_r))
    print(paste0("trained ", Best_model_param_desc_fd_r)) 
    print(paste0("training time: ", round(time,2), " minutes at ", end.time)) 
#   }
# }
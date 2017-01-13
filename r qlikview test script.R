#simple script to test statconn R qlikview
library(h2o)
h2o.init(nthreads=7, max_mem_size="20G")
h2o.removeAll()
df<-read.csv(paste0(getwd(),'/data/ExportFile.csv'))

names(df)[1]<-"ACT"

data_full<-df[,]


data_full$Month<-as.factor(data_full$Month)


YearStart<-2004

RowStart<-1

YearEnd<-2014

RowEnd2014<-3848
RowEnd2015<-4213

if(YearEnd==2014) {RowEnd<-RowEnd2014}else if (YearEnd==2015){RowEnd<-RowEnd2015}



data<-data_full[(RowStart:RowEnd),]

data.h2o<-as.h2o(data,destination_frame = "data.h2o")

data_full.h2o<-as.h2o(data_full, destination_frame = "data_full.h2o")

predictors<-c(3:(ncol(data)))   #3:12
predictors

response <- 1
response

data.split<-h2o.splitFrame(data.h2o, ratios=c(0.8), destination_frames = c("train_r.h2o","valid_r.h2o") )

train_r.h2o <- data.split[[1]]
str(train_r.h2o)

valid_r.h2o <- data.split[[2]]


#################################################################################################
########### DNN 
#############################################################################
mid<-paste('R_ADHOC','Rand','DNN','Data_Since',YearStart,'Data_till',YearEnd,'year_end',sep = '_')

best_model_r <- h2o.deeplearning(
  model_id = mid,
  training_frame=train_r.h2o,
  validation_frame=valid_r.h2o, 
  x=predictors, 
  y=response,
  epochs=5000,
  stopping_metric="MSE",
  hidden=c(800,800),
  #  l1=(1e-6),
  #  l2=(1e-6),
  activation =("Maxout")
)



pr.h2o_dnn_fd_r<- h2o.predict(best_model_r, data_full.h2o[,predictors])
pr.dnn_fd.df_r<-  as.data.frame(pr.h2o_dnn_fd_r)
pr.dnn_fd_r<-pr.dnn_fd.df_r[[1]]


#savedDNN_Rand_R<-h2o.saveModel(best_model_r, path = "H2O_Models", force = TRUE)
#savedDNN_Rand_R_POJO<-h2o.download_pojo(best_model_r, path = "POJO", getjar = TRUE)
######################################################################################################
################# H2O GLM
# mid<-paste('Rand','GLM','Data_Since',YearStart,'Data_till',YearEnd,'year_end',sep = '_')
# mid
# glm_model_r <- h2o.glm(model_id = mid, training_frame=train_r.h2o, validation_frame=valid_r.h2o, x=predictors, y=response,max_iterations=5000)
# 
# #savedGLM_Rand<-h2o.saveModel(glm_model_r, path = "H2O_Models", force = TRUE)
# #savedGLM_Rand_POJO<-h2o.download_pojo(glm_model_r, path = "POJO", getjar = TRUE)
# 
# #summary(glm_model_r)
# 
# #scoring_history_glm_r <- as.data.frame(glm_model_r@model$scoring_history)
# #summary(scoring_history_glm_r)
# 
# pr.h2o_glm_fd_r<- h2o.predict(glm_model_r, data_full.h2o[,predictors])
# pr.glm_fd.df_r<-  as.data.frame(pr.h2o_glm_fd_r)
# pr.glm_fd_r<-pr.glm_fd.df_r[[1]]

####################################################################################################
############# Write to CSV
####################################################################################################
#best_params_r <- best_model_r@allparameters
best_params_r <- glm_model_r@allparameters
Best_model_param_desc_fd_r<-paste('R_ADHOC','Rand ','Data Starts at ',YearStart,' beginning ','Data Ends at ',YearEnd,' end',best_params_r$activation,' ',toString(best_params_r$hidden),' ','l1=',best_params_r$l1,' l2=',best_params_r$l2,'_fd.csv',sep="")
Best_model_param_desc_fd_r

export_df_pr_fd_r<-cbind(#predict_dnn_r=pr.dnn_fd_r,
                         predict_glm=pr.glm_fd_r, data_full)

#write.csv(export_df_pr_fd_r,file=Best_model_param_desc_fd_r,row.names = F)


write.csv(export_df_pr_fd_r, file= paste0(getwd(),'/results/ExportFile_GLM.csv'), row.names=FALSE)
#write.csv(export_df_pr_fd_r, file='"&CurrentPath&"/results/ExportFile_GLM.csv', row.names=FALSE)
#h2o.shutdown(prompt = FALSE)


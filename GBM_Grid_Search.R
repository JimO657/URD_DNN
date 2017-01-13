#Saved on WorkS
set.seed(500)     
rm(list = ls())
library(h2o)
h2o.init(nthreads=-1, max_mem_size='50G')
h2o.removeAll()
df<-read.csv('data/ExportFileR.csv')

names(df)[1]<-'ACT'

data_full<-df[,]

data_full$Month1<-as.factor(data_full$Month1)

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

head(data.h2o)


RS<-RowEnd+1
RE<-nrow(data_full.h2o)

test<-data_full[(RS:RE),]

test.h2o<-as.h2o(test,destination_frame = 'test.h2o')
head(test.h2o)
tail(test.h2o)
  
  
head(test.h2o)


predictors<-c(3:(ncol(data)))
predictors
response <- 1
response
data.split<-h2o.splitFrame(data.h2o, ratios=c(0.8), destination_frames = c('train_r.h2o','valid_r.h2o') )
train_r.h2o <- data.split[[1]]
str(train_r.h2o)

valid_r.h2o <- data.split[[2]]
#'''''''''''''' ##Set Mid ??????????????????????????? '''''''''''' # 
#mid<-paste('R_ADHOC','Rand','GBM','Data_Since',YearStart,'Data_till',YearEnd,'year_end',sep = '_')

############################################################################
############## Grid Search Hyper Param
############################################################################
## Hyper-Parameter Search

## Construct a large Cartesian hyper-parameter space
ntrees_opts <- c(10000) ## early stopping will stop earlier
max_depth_opts <- seq(1,20)
min_rows_opts <- c(1,5,10,20,50,100)
learn_rate_opts <- seq(0.001,0.01,0.001)
sample_rate_opts <- seq(0.3,1,0.05)
col_sample_rate_opts <- seq(0.3,1,0.05)
col_sample_rate_per_tree_opts = seq(0.3,1,0.05)
#nbins_cats_opts = seq(100,10000,100) ## no categorical features in this dataset

hyper_params = list( ntrees = ntrees_opts, 
                     max_depth = max_depth_opts, 
                     min_rows = min_rows_opts, 
                     learn_rate = learn_rate_opts,
                     sample_rate = sample_rate_opts,
                     col_sample_rate = col_sample_rate_opts,
                     col_sample_rate_per_tree = col_sample_rate_per_tree_opts
                     #,nbins_cats = nbins_cats_opts
)


hyper_params

mid<-paste('GRID_GBM','Data_Since','2006','Data_till',YearEnd,'year_end',sep = '_')


start.time <- Sys.time()

#gbm_model_r <- h2o.gbm(model_id = mid, training_frame=train_r.h2o, validation_frame=valid_r.h2o, x=predictors, y=response, ntrees=500, learn_rate=0.01, score_each_iteration = TRUE)

grid_gbm <- h2o.grid(
  "gbm",
  grid_id= mid, 
  training_frame=train_r.h2o,
  validation_frame=valid_r.h2o, 
  x=predictors, 
  y=response,
  score_each_iteration = TRUE,
  #############################
  ### Early Stopping 
  
  distribution="gaussian", ## best for MSE loss, but can try other distributions ("laplace", "quantile")
  ## stop as soon as mse doesn't improve by more than 0.1% on the validation set, 
  ## for 2 consecutive scoring events
  stopping_rounds = 5,
  stopping_tolerance = 1e-3,
  stopping_metric = "MSE",
  score_tree_interval = 100, ## how often to score (affects early stopping)
  seed = 123456, ## seed to control the sampling of the Cartesian hyper-parameter space
  ############
  hyper_params=hyper_params
)

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

grid_gbm



# Let's see which model had the lowest validation error:1

## Find the best model and its full set of parameters (clunky for now, will be improved)

# scores <- cbind(as.data.frame(unlist((lapply(grid@model_ids, function(x) 
# { h2o.confusionMatrix(h2o.performance(h2o.getModel(x),valid=T))$Error[8] })) )), unlist(grid@model_ids))
# 
# names(scores) <- c("misclassification","model")
# 
# sorted_scores <- scores[order(scores$misclassification),]
# 
# head(sorted_scores)
# 
# best_model <- h2o.getModel(as.character(sorted_scores$model[1]))
# 
# print(best_model@allparameters)
# 
# best_err <- sorted_scores$misclassification[1]
# 
# print(best_err)





grid_gbm_models <- lapply(grid_gbm@model_ids, 
                          function(mid) {
                                         model = h2o.getModel(mid)
                                        }
                          )





################################################################
###### Check the performance of the models ##############
## Check performance of the first model
h2o.performance(model = grid_gbm_models[[1]], newdata = test.h2o)


grid_gbm_Valid_MSEs <- sapply(grid_gbm_models, 
                              function(modeli) {
                                modeli@model$validation_metrics@metrics$MSE
                              }
)

plot(grid_gbm_Valid_MSEs)

#### Check the models one by one ##

print(paste('There are ',length(grid_gbm_models),' models in DNN_GRID'))

CheckGrid <- function(){
  ans<-'Y'
  i<-1
  j<-1
  while(ans!='q' & i<=length(grid_gbm_models))
  {
    print(paste('Model Number = ',i, ' of ',length(grid_gbm_models)))
    print(grid_gbm_models[[i]])
    
    print(h2o.performance(model = grid_gbm_models[[i]], newdata = test.h2o))
    
    time <- grid_gbm_models[[i]]@model$run_time/1000/60
    print(paste0("training time is: ", round(time,2), ' minutes.'))
    
    ans<-readline(prompt = "Press 'n' for next model, 'b' for previous, q to quit: ")
    if(ans=='n'){j<-1}else if(ans=='b'){j<- (-1)}
    i<-i+j
  }
}
CheckGrid()

############ Plot of the perfomrnace tests on Test Data

Best_test_MSE<-h2o.performance(model = grid_gbm_models[[24]], newdata = test.h2o)
Best_test_MSE

s<-grid_gbm_models[[24]]@model_id
s2<-substr(s, nchar(s)-2+1, nchar(s))
if(substr(s2, 1, 1)=='_'){s3<-substr(s2,2,2)} else s3<-s2
s3

grid_gbm_Test_MSEs <- sapply(grid_gbm_models, 
                              function(modeli) {
                                #modeli@model$validation_metrics@metrics$MSE
                                h2o.performance(model = modeli, newdata = test.h2o)@metrics$MSE
                              }
)

grid_gbm_Test_MNOs <- sapply(grid_gbm_models, 
                             function(modeli) {
                               #modeli@model$validation_metrics@metrics$MSE
                               #h2o.performance(model = modeli, newdata = test.h2o)@metrics$MSE
                               s<-modeli@model_id
                               s2<-substr(s, nchar(s)-2+1, nchar(s))
                               if(substr(s2, 1, 1)=='_'){s3<-substr(s2,2,2)} else s3<-s2
                               s3
                             }
)
plot(grid_gbm_Test_MSEs)
min((grid_gbm_Test_MSEs))
plot(grid_gbm_Test_MNOs,grid_gbm_Valid_MSEs)

plot(grid_gbm_Test_MNOs,grid_gbm_Test_MSEs)

############################################################

grid_gbm_model_r <- grid_gbm_models[[1]]
grid_gbm_model_r
grid_gbm_params <- grid_gbm_model_r@allparameters
grid_gbm_params$activation
grid_gbm_params$hidden
grid_gbm_params$l1
grid_gbm_params$l2
grid_gbm_params$input_dropout_ratio

grid_gbm_model_param_desc<-paste(grid_gbm_params$activation,' ',toString(grid_gbm_params$hidden),' ','l1=',grid_gbm_params$l1,' l2=',grid_gbm_params$l2,'.csv',sep="")
grid_gbm_model_param_desc
#################################################################################################
########### SAve the best model from grid
saved_GBM_best<-h2o.saveModel(grid_gbm_model_r, path = '/0MyDataBases/7R/ADHOC_Qlikview/H2O_Models', force = TRUE)            
saved_GBM_best


###  BEST !!!! Test MSE of 18.9559 !!!!!!!
saved_GBM_best_T_MSE<-h2o.saveModel(grid_gbm_models[[24]], path = '/0MyDataBases/7R/ADHOC_Qlikview/H2O_Models', force = TRUE)            
saved_GBM_best_T_MSE # = "C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models\\Grid_DeepLearning_train_r.h2o_model_R_1463494899000_2_model_23"


##################################################################################



# #  ''''''''''''''' #LOAD Models '''''''''''''
 #    
 glm_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models\\R_ADHOC_GLM_Data_Since_2006_Data_till_2014_year_end') 
# #    WriteLog("GLM Model Loaded "&Runtime&". "&Runtime&".")
    
 gbm_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models\\R_ADHOC_GBM_Data_Since_2006_Data_till_2014_year_end') 
#    WriteLog("GBM Model Loaded "&Runtime&".")  

 dnn_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models\\R_ADHOC_DNN_Data_Since_2006_Data_till_2014_year_end') 
# ##    WriteLog("DNN Model Loaded "&Runtime&".")
    
 best_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models\\0Best_R_ADHOC_GBM_Data_Since_2006_Data_till_2014_year_end') 
# #    WriteLog("BEST Model Loaded "&Runtime&".")
 #  '''''''''''''''''''''''''''''''''''''''''''' ## 
 #  ''''''''' ##Create Predictions form loaded models '''''''''''''''' ##       
 pr.glm<- h2o.predict(glm_model_r, data_full.h2o[,predictors]) 
 pr.glm_fd_r<- (as.data.frame(pr.glm$pred))[,1] 
# #    WriteLog("GLM Model Predicted "&Runtime&".")
 #    
 pr.gbm<- h2o.predict(gbm_model_r, data_full.h2o[,predictors]) 
 pr.gbm_fd_r<- (as.data.frame(pr.gbm$pred))[,1] 
# #    WriteLog("GBM Model Predicted "&Runtime&".")
 #    
  pr.dnn<- h2o.predict(dnn_model_r , data_full.h2o[,predictors])
  pr.dnn_fd_r<-as.data.frame(pr.dnn$pred)[,1]
# #    WriteLog("DNN Model predicted at "&Runtime&". ")  
 #              
  pr.best<- h2o.predict(best_model_r, data_full.h2o[,predictors])
  pr.best_fd_r<-as.data.frame(pr.best$pred)[,1]
# #    WriteLog("DNN best model predicted at "&Runtime&". ")
 #    
  pr.grid_gbm<- h2o.predict(grid_gbm_model_r, data_full.h2o[,predictors])
  pr.grid_gbm_fd_r<-as.data.frame(pr.grid_gbm$pred)[,1]
  #    
#''''''''''''''''''' ##Write out 
 export_df_pr_fd_r<-cbind(predict_GRID_gbm=pr.grid_gbm_fd_r,
                          predict_BEST=pr.best_fd_r,predict_GLM=pr.glm_fd_r,predict_GBM=pr.gbm_fd_r,predict_DNN=pr.dnn_fd_r, data_full)
#
 write.csv(export_df_pr_fd_r, file='results/ExportFile_PRED_COMBINED.csv', row.names=FALSE)
 #    WriteLog("DNN CSV written at "&Runtime&". ")   
######## Plot
setwd('/0MyDataBases/7R/ADHOC_Qlikview/plots') 
png('Score_Plot_GRID_GBM.png')
plot(grid_gbm_model_r,metric='MSE')
title(main='GRID_GBM',  col.main='black', line='3', font.main=4)
dev.off()
setwd('/0MyDataBases/7R/ADHOC_Qlikview')
######################

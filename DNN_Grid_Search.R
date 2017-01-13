#Saved on Workstation

set.seed(500)     
rm(list = ls())
library(h2o)

#h2o.shutdown(prompt = FALSE)
h2o.init(nthreads= 7, max_mem_size='35G')
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
##################################################


data<-data_full[(RowStart:RowEnd),]
data.h2o<-as.h2o(data,destination_frame = 'data.h2o')
data_full.h2o<-as.h2o(data_full, destination_frame = 'data_full.h2o')

head(data.h2o)


RS<-RowEnd+1


de<- paste0(YearEnd,'-Dec-31')

dee<- paste0('2016-Jun-30')

RET<-which(data_full$Date1 == dee)
RET


test<-data_full[(RS:RET),]

test.h2o<-as.h2o(test,destination_frame = 'test.h2o')
head(test.h2o)
tail(test.h2o)
  
  
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
## Search a random subset of these hyper-parmameters (max runtime and max models are enforced, and the search will stop after we don't improve much over the best 5 random models)


search_criteria = list(#strategy = "Cartesian",
                       strategy = "RandomDiscrete", 
#                       max_runtime_secs = 600, 
                       max_models = 500, 
                       stopping_metric = "MSE", 
                       stopping_tolerance = (1e-6), 
                       stopping_rounds = 20, 
                       seed = 123456)
   


######################################################################################


######################################################################################
hyper_params <- list(
  hidden=list(c(800,800),c(5000),c(800,800,800),c(200,200,200,200),c(500,500,500)),
  #  input_dropout_ratio=c(0,0.05),
  l1=c(0,1e-4,1e-6,1e-12),
  l2=c(0,1e-4,1e-6,1e-12),
  activation =c("MaxoutWithDropout","Rectifier","RectifierWithDropout","Tanh","TanhWithDropout","Maxout")
  #  hidden_dropout_ratios = c(0.0015,0.0015)
  #  rate=c(0.01,0.02),
  #  rate_annealing=c(1e-8,1e-7,1e-6)
)
hyper_params

mid<-paste('R_ADHOC','Rand','grid_dnn','Data_Since',YearStart,'Data_till',YearEnd,'year_end',sep = '_')


start.time <- Sys.time()

#dnn_model_r <- h2o.deeplearning(model_id = mid, training_frame=train_r.h2o, validation_frame=valid_r.h2o, x=predictors, y=response, epochs=5000, hidden=c(5000),    activation =('MaxoutWithDropout'), l1=(1e-6),l2=(1e-6),stopping_metric= 'MSE',stopping_rounds=20,stopping_tolerance=(1e-6))


grid_dnn <- h2o.grid(
  "deeplearning",
  grid_id= mid, 
  training_frame=train_r.h2o,
  validation_frame=valid_r.h2o, 
  x=predictors, 
  y=response,
  epochs= 5000,
  stopping_metric="MSE",#"misclassification",
    stopping_tolerance=1e-6,        ## stop when logloss does not improve by >=1% for 2 scoring events
    stopping_rounds=20,
  #  score_validation_samples=10000, ## downsample validation set for faster scoring
  #  score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
  #  adaptive_rate=F,                ## manually tuned learning rate
  #  momentum_start=0.5,             ## manually tuned momentum
  #  momentum_stable=0.9, 
  #  momentum_ramp=1e7, 
  #  activation=c("Rectifier"),
  #  max_w2=10, ## can help improve stability for Rectifier
  hyper_params = hyper_params
  ,search_criteria = search_criteria
)

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

grid_dnn



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





grid_dnn_models <- lapply(grid_dnn@model_ids, 
                          function(mid) {
                                         model = h2o.getModel(mid)
                                        }
)
################################################################
###### Check the performance of the models ##############
## Check performance of the first model


grid_dnn_Valid_MSEs <- sapply(grid_dnn_models, 
                              function(modeli) {
                                modeli@model$validation_metrics@metrics$MSE
                              }
)

plot(grid_dnn_Valid_MSEs)

#### Check the models one by one ##

print(paste('There are ',length(grid_dnn_models),' models in DNN_GRID'))

# GetNumber<-function(modeli){
#   s<-modeli@model_id
#   s2<-substr(s, nchar(s)-2+1, nchar(s))
#   if(substr(s2, 1, 1)=='_'){s3<-substr(s2,2,2)} else s3<-s2
#   as.integer(s3)
# }

CheckGrid <- function(){
  ans<-'Y'
  i<-1
  j<-1
  while(ans!='q' & i<=length(grid_dnn_models))
  {
    print(paste('Model Number = ', i  , ' of ',length(grid_dnn_models)))
    print(grid_dnn_models[[i]])
    
    print(h2o.performance(model = grid_dnn_models[[i]], newdata = test.h2o))
    
    time <- grid_dnn_models[[i]]@model$run_time/1000/60
    print(paste0("training time is: ", round(time,2), ' minutes.'))
    
    ans<-readline(prompt = "Press 'n' for next model, 'b' for previous, q to quit: ")
    if(ans=='n'){j<-1}else if(ans=='b'){j<- (-1)}
    i<-i+j
  }
}


############ Find the best models by MSE on Test data

# MSE=as.double(h2o.performance(model = grid_dnn_models[[1]], newdata = test.h2o)@metrics$MSE)
# MSE

grid_dnn_Test_MSEs <- sapply(grid_dnn_models, 
                             function(modeli) {
                               #modeli@model$validation_metrics@metrics$MSE
                               MSE=as.double(h2o.performance(model = modeli, newdata = test.h2o)@metrics$MSE)
                             }
)


# grid_dnn_Test_MNOs <- sapply(grid_dnn_models, 
#                              function(modeli) {
#                                MNOs=GetNumber(modeli) 
#                              }
# )
# grid_dnn_Test_MNOs

grid_dnn_Test<-as.data.frame(cbind(MNOs=c(1:length(grid_dnn_Test_MSEs)),MSE=grid_dnn_Test_MSEs))
grid_dnn_Test

plot(grid_dnn_Test)

best_no<- grid_dnn_Test$MNOs[which(grid_dnn_Test$MSE==min((grid_dnn_Test$MSE)))]
best_no

time <- grid_dnn_models[[best_no]]@model$run_time/1000/60
print(paste0("training time for best model is: ", round(time,2), ' minutes.'))
print(paste0("Total training time for the grid is: ", round(time.taken,2), ' days'))
time.taken


plot(grid_dnn_models[[best_no]])


Best_test_MSE<-h2o.performance(model = grid_dnn_models[[best_no]], newdata = test.h2o)
Best_test_MSE

First_test_MSE<-h2o.performance(model = grid_dnn_models[[1]], newdata = test.h2o)
First_test_MSE

############################################################
grid_dnn_model_1 <- grid_dnn_models[[1]]
grid_dnn_model_1
grid_dnn_model_r <- grid_dnn_models[[best_no]]
grid_dnn_model_r
grid_dnn_params <- grid_dnn_model_r@allparameters
grid_dnn_params
# ############################################################
# 
grid_dnn_params <- grid_dnn_model_r@allparameters
grid_dnn_params$activation
grid_dnn_params$hidden
grid_dnn_params$l1
grid_dnn_params$l2
grid_dnn_params$input_dropout_ratio

grid_dnn_model_param_desc<-paste(grid_dnn_params$activation,' ',toString(grid_dnn_params$hidden),' ','l1=',grid_dnn_params$l1,' l2=',grid_dnn_params$l2,'.csv',sep="")
grid_dnn_model_param_desc

ttt<-paste0('Time to Train = ',round(time,2), ' minutes')

fileConn<-file("results_2015/GRID_DNN_BEST_MODEL_PARAMS.txt")
writeLines(c("GRID_DNN_BEST_MODEL_PARAMS",ttt, grid_dnn_model_param_desc), fileConn)
close(fileConn)

#################################################################################################
########### SAve the best model from grid
saved_DNN_First<-h2o.saveModel(grid_dnn_model_1, path = '/0MyDataBases/7R/ADHOC_Qlikview/H2O_Models_2015', force = TRUE)            
saved_DNN_First
saved_DNN_GRID<-h2o.saveModel(grid_dnn_model_r, path = '/0MyDataBases/7R/ADHOC_Qlikview/H2O_Models_2015', force = TRUE)            
saved_DNN_GRID

##################################################################################



# #  ''''''''''''''' #LOAD Models '''''''''''''
#    
glm_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models_2015\\R_ADHOC_GLM_Data_Since_2006_Data_till_2014_year_end') 
# #    WriteLog("GLM Model Loaded "&Runtime&". "&Runtime&".")

gbm_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models_2015\\R_ADHOC_GBM_Data_Since_2006_Data_till_2014_year_end') 
#    WriteLog("GBM Model Loaded "&Runtime&".")  

dnn_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models_2015\\R_ADHOC_DNN_Data_Since_2006_Data_till_2014_year_end') 
# ##    WriteLog("DNN Model Loaded "&Runtime&".")

best_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models_2015\\R_ADHOC_Rand_grid_dnn_Data_Since_2006_Data_till_2014_year_end_model_103') 
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
pr.grid_dnn<- h2o.predict(grid_dnn_model_r, data_full.h2o[,predictors])
pr.grid_dnn_fd_r<-as.data.frame(pr.grid_dnn$pred)[,1]
#    
#''''''''''''''''''' ##Write out 
export_df_pr_fd_r<-cbind(predict_GRID_DNN=pr.grid_dnn_fd_r,
                         predict_BEST=pr.best_fd_r,predict_GLM=pr.glm_fd_r,predict_GBM=pr.gbm_fd_r,predict_DNN=pr.dnn_fd_r, data_full)
#
write.csv(export_df_pr_fd_r, file='results_2015/ExportFile_PRED_COMBINED.csv', row.names=FALSE)
#    WriteLog("DNN CSV written at "&Runtime&". ")   
######## Plot
setwd('/0MyDataBases/7R/ADHOC_Qlikview/plots') 
png('Score_Plot_grid_dnn.png')
plot(grid_dnn_model_r,metric='MSE')
title(main='grid_dnn',  col.main='black', line='3', font.main=4)
dev.off()
setwd('/0MyDataBases/7R/ADHOC_Qlikview')
######################

#
CheckGrid()

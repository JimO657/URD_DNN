#Saved on WorkS
# Source: https://github.com/h2oai/h2o-3/blob/master/h2o-docs/src/product/tutorials/GridSearch.md

set.seed(500)     
rm(list = ls())
library(h2o)
h2o.init(nthreads=-1, max_mem_size='15G')
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
dee<- paste0('2016-Jun-30')

RET<-which(data_full$Date1 == dee)
RET


test<-data_full[(RS:RET),]

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

mid<-paste('GRID_GBM','Data_Since',YearStart,'Data_till',YearEnd,'year_end',sep = '_')


start.time <- Sys.time() 


###################################################################
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


## Search a random subset of these hyper-parmameters (max runtime and max models are enforced, and the search will stop after we don't improve much over the best 5 random models)
search_criteria = list(strategy = "RandomDiscrete", 
                       #max_runtime_secs = 600, 
                       max_models = 100, 
                       stopping_metric = "MSE", 
                       stopping_tolerance = 1e-6, 
                       stopping_rounds = 20, 
                       seed = 123456)



grid_gbm <- h2o.grid(
  "gbm",
  grid_id= mid, 
  training_frame=train_r.h2o,
  validation_frame=valid_r.h2o, 
  x=predictors, 
  y=response,
#  stopping_metric="MSE",#"misclassification",
  score_each_iteration = TRUE,
  nfolds = 0,
############
  distribution="gaussian", ## best for MSE loss, but can try other distributions ("laplace", "quantile")
  ## stop as soon as mse doesn't improve by more than stopping_tolerance% on the validation set, 
  ## for stopping_rounds consecutive scoring events
  stopping_metric= 'MSE',
  stopping_rounds=20,
  stopping_tolerance=1e-6, 
  score_tree_interval = 100, ## how often to score (affects early stopping)
  #seed = 123456, ## seed to control the sampling of the Cartesian hyper-parameter space
  hyper_params = hyper_params,
#### Enable for RANDOM GRID SEARCH ################
  search_criteria = search_criteria
)

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken
######


#######################################################################
gbm.sorted.grid <- h2o.getGrid(grid_id = mid, sort_by = "mse")
print(gbm.sorted.grid)

best_model <- h2o.getModel(gbm.sorted.grid@model_ids[[1]])
summary(best_model)

scoring_history <- as.data.frame(best_model@model$scoring_history)
plot(scoring_history$number_of_trees, scoring_history$training_MSE, type="p") #training mse
points(scoring_history$number_of_trees, scoring_history$validation_MSE, type="l") #validation mse

## get the actual number of trees
ntrees <- best_model@model$model_summary$number_of_trees
print(ntrees)

##################################################################
########### My part #############


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

print(paste('There are ',length(grid_gbm_models),' models in gbm_GRID'))
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
  while(ans!='q' & i<=length(grid_gbm_models))
  {
    print(paste('Model Number = ', i  , ' of ',length(grid_gbm_models)))
    print(grid_gbm_models[[i]])
    
    print(h2o.performance(model = grid_gbm_models[[i]], newdata = test.h2o))
    
    time <- grid_gbm_models[[i]]@model$run_time/1000/60
    print(paste0("training time is: ", round(time,2), ' minutes.'))
    
    ans<-readline(prompt = "Press 'n' for next model, 'b' for previous, q to quit: ")
    if(ans=='n'){j<-1}else if(ans=='b'){j<- (-1)}
    i<-i+j
  }
}


############ Find the best models by MSE on Test data
############ Find the best models by MSE on Test data

# MSE=as.double(h2o.performance(model = grid_gbm_models[[1]], newdata = test.h2o)@metrics$MSE)
# MSE

grid_gbm_Test_MSEs <- sapply(grid_gbm_models, 
                             function(modeli) {
                               #modeli@model$validation_metrics@metrics$MSE
                               MSE=as.double(h2o.performance(model = modeli, newdata = test.h2o)@metrics$MSE)
                             }
)


# grid_gbm_Test_MNOs <- sapply(grid_gbm_models, 
#                              function(modeli) {
#                                MNOs=GetNumber(modeli) 
#                              }
# )
# grid_gbm_Test_MNOs

grid_gbm_Test<-as.data.frame(cbind(MNOs=c(1:length(grid_gbm_Test_MSEs)),MSE=grid_gbm_Test_MSEs))
grid_gbm_Test

plot(grid_gbm_Test)

best_no<- grid_gbm_Test$MNOs[which(grid_gbm_Test$MSE==min((grid_gbm_Test$MSE)))]
best_no

time <- grid_gbm_models[[best_no]]@model$run_time/1000/60
print(paste0("training time is: ", round(time,2), ' minutes.'))



plot(grid_gbm_models[[best_no]])


Best_test_MSE<-h2o.performance(model = grid_gbm_models[[best_no]], newdata = test.h2o)
Best_test_MSE


############################################################
grid_gbm_model_1 <- grid_gbm_models[[1]]
grid_gbm_model_1
grid_gbm_model_r <- grid_gbm_models[[best_no]]
grid_gbm_model_r
grid_gbm_params <- grid_gbm_model_r@allparameters
grid_gbm_params
# ############################################################
# 
grid_gbm_params <- grid_gbm_model_r@allparameters

grid_gbm_model_param_desc<-paste(
  'ntrees=', grid_gbm_params$ntrees, 
  ' max_depth=', grid_gbm_params$max_depth, 
  ' min_rows=', grid_gbm_params$min_rows, 
  ' learn_rate=', grid_gbm_params$learn_rate,
  ' sample_rate=', grid_gbm_params$sample_rate,
  ' col_sample_rate=', grid_gbm_params$col_sample_rate ,
  ' col_sample_rate_per_tree=', grid_gbm_params$col_sample_rate_per_tree
  ,'.csv',sep="")
grid_gbm_model_param_desc
ttt<-paste0('Time to Train = ',round(time,2), ' minutes')

fileConn<-file("results_2015/GRID_GBM_BEST_MODEL_PARAMS.txt")
writeLines(c("GRID_GBM_BEST_MODEL_PARAMS", ttt , grid_gbm_model_param_desc), fileConn)
close(fileConn)

#################################################################################################
########### SAve the best model from grid
saved_gbm_best<-h2o.saveModel(grid_gbm_model_1, path = '/0MyDataBases/7R/ADHOC_Qlikview/H2O_Models_2015', force = TRUE)            
saved_gbm_best
saved_gbm_GRID<-h2o.saveModel(grid_gbm_model_r, path = '/0MyDataBases/7R/ADHOC_Qlikview/H2O_Models_2015', force = TRUE)            
saved_gbm_GRID

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
pr.grid_gbm<- h2o.predict(grid_gbm_model_r, data_full.h2o[,predictors])
pr.grid_gbm_fd_r<-as.data.frame(pr.grid_gbm$pred)[,1]
#    
#''''''''''''''''''' ##Write out 
export_df_pr_fd_r<-cbind(predict_GRID_GBM=pr.grid_gbm_fd_r,
                         predict_BEST=pr.best_fd_r,predict_GLM=pr.glm_fd_r,predict_GBM=pr.gbm_fd_r,predict_DNN=pr.dnn_fd_r, data_full)
#
write.csv(export_df_pr_fd_r, file='results_2015/ExportFile_PRED_COMBINED.csv', row.names=FALSE)
#    WriteLog("DNN CSV written at "&Runtime&". ")   
######## Plot
setwd('/0MyDataBases/7R/ADHOC_Qlikview/plots') 
png('Score_Plot_grid_gbm.png')
plot(grid_gbm_model_r,metric='MSE')
title(main='grid_gbm',  col.main='black', line='3', font.main=4)
dev.off()
setwd('/0MyDataBases/7R/ADHOC_Qlikview')
######################
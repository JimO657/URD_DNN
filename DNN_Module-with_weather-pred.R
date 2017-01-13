#################	
library(h2o)

#h2o.init()
h2o.init(nthreads=70, max_mem_size='80G')
##WriteLog("DNN H2O started at "&Runtime&". ")
h2o.removeAll()
df<-read.csv('data_2015/ExportFileR.csv')
##WriteLog("DNN datat read at "&Runtime&". ")           
names(df)[1]<-'ACT'

data_full<-df[,]

data_full$Month1<-as.factor(data_full$Month1)


######################################             
#         #### Choose Start end end years to use for training 
##WriteLog("DNN about to set YearStart to mrvStartYear "&Runtime&". ")			            
# YearStart<-"&mrvStartYear 
         YearStart<-2006
##WriteLog("DNN about to set YearEnd to mrvEndYear "&Runtime&". ")			            
#YearEnd<-"&mrvEndYear 
         YearEnd<-2014


ds<- paste0(YearStart,'-Jan-01')

RowStart<-which(data_full$Date1 == ds)
RowStart

de<- paste0(YearEnd,'-Dec-31')

RowEnd<-which(data_full$Date1 == de)
RowEnd
##WriteLog("DNN row numbers set "&Runtime&". ")			            
##################################################
################################################

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
######## Start DNN ####### 
mid<-paste('R_ADHOC','DNN','Data_Since',YearStart,'Data_till',YearEnd,'year_end',sep = '_')
##WriteLog("DNN Training started at "&Runtime&". ")
#		dnn_model_r <- h2o.deeplearning(model_id = mid, training_frame=train_r.h2o, validation_frame=valid_r.h2o, x=predictors, y=response, epochs=5000, hidden=c(800,800), activation =('Maxout'), l1=0, l2=0)
#dnn_model_r <- h2o.deeplearning(model_id = mid, training_frame=train_r.h2o, validation_frame=valid_r.h2o, x=predictors, y=response, epochs=5000, hidden=c(800,800), activation =('Maxout'), l1=0, l2=0,stopping_rounds=25,stopping_metric= 'MSE',stopping_tolerance=1e-6)
############  remove early stopping
# Start the clock!
ptm <- proc.time()
dnn_model_r <- h2o.deeplearning(model_id = mid, training_frame=train_r.h2o, validation_frame=valid_r.h2o, x=predictors, y=response, epochs=5000, hidden=c(800,800), activation =('Tanh'), l1=0, l2=0, stopping_rounds=5, stopping_metric= 'MSE',stopping_tolerance=1e-6)

#dnn_model_r <- h2o.deeplearning(model_id = mid, training_frame=train_r.h2o, validation_frame=valid_r.h2o, x=predictors, y=response, epochs=500, hidden=c(800,800,800), activation =('Tanh'), l1=0, l2=0,stopping_rounds=0 ) #20,stopping_metric= 'MSE',stopping_tolerance=1e-6)
# Stop the clock
timet<-proc.time() - ptm
timet


#####################################################################

setwd('plots_2015-R') 
png('Score_Plot_DNN.png')
plot(dnn_model_r,metric='rmse')
title(main='DNN',  col.main='black', line='3', font.main=4)
dev.off()
setwd('/home/norayr/1MyDataBases-short/100deepwater-master/ADHOC_Qlikview')
#WriteLog("DNN plot saved at "&Runtime&". ")


#########Enable Cross Validation ########     
#           nfolds <- 5
#		dnn_model_r <- h2o.deeplearning(model_id = mid, training_frame=train_r.h2o, validation_frame=valid_r.h2o, x=predictors, y=response, epochs=5000, hidden=c(800,800), activation =('Maxout'), l1=0, l2=0,nfolds = nfolds, fold_assignment = 'Modulo', keep_cross_validation_predictions = TRUE)

##WriteLog("DNN Training finished at "&Runtime&". ")
########## End DNN

######Save Model and POJO ########
savedDNN_Rand<-h2o.saveModel(dnn_model_r, path = 'H2O_Models_2015-R', force = TRUE)            
#WriteLog("DNN model saved at "&Runtime&". ")
#savedDNN_Rand_POJO<-h2o.download_pojo(dnn_model_r, path = 'POJO_2015-R', get_jar = TRUE)
#WriteLog("DNN POJO saved at "&Runtime&". ")

#####################################################################
##### Predict on original data

pr.dnn<- h2o.predict(dnn_model_r, data_full.h2o[,predictors])
pr.dnn_fd_r<-as.data.frame(pr.dnn$pred)[,1]
#WriteLog("DNN Model Predicted "&Runtime&".")            
#####################################################################


#####################################################################
##### Predict on WEATHER data
dfw<-read.csv('data_2015/ExportFileWeather_2010.csv')

names(dfw)[1]<-'ACT'

data_fullw<-dfw[,]

data_fullw$Month1<-as.factor(data_fullw$Month1)

#'''''''''''''' Setting variables  ''''''''''''''''''''''''''''''''''''''' 

#WriteLog("Predict about to set variables "&Runtime&". ")

data_fullw.h2o<-as.h2o(data_fullw, destination_frame = 'data_fullw.h2o')

# WriteLog("#Weather Predict data_full Loaded "&Runtime&".")
predictors<-c(3:(ncol(data_fullw)))
predictors
response <- 1
response

#'''''''''''''''' LOAD Models '''''''''''''  

#WriteLog("#Weather about to load DNN Model "&Runtime&".")
#dnn_model_r <- h2o.loadModel('C:\\0MyDataBases\\7R\\ADHOC_Qlikview\\H2O_Models_2015\\R_ADHOC_DNN_Data_Since_"&DNNStartYear&"_Data_till_"&DNNEndYear&"_year_end')
#WriteLog("#Weather DNN Model Loaded "&Runtime&".")

pr.dnn_w<- h2o.predict(dnn_model_r, data_fullw.h2o[,predictors])
pr.dnn_fd_r_w<-as.data.frame(pr.dnn_w$pred)[,1]
         


print("Time for training:")
print(timet)







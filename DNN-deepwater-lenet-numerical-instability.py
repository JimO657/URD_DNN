# ### Lenet mxnet version
#
# Using H2ODeepLearningEstimator on line 194 works fine
#
# # When using the straight deepwater estimator (line 203)
# # I get the following error no matter what I do with the predictors, activation function, regularization etc... :
# ####
# # Trying to predict with an unstable model.
# # Job was aborted due to observed numerical instability (exponential growth).
# # Either the weights or the bias values are unreasonably large or lead to large activation values.
# # Try a different network architecture, a bounded activation function (tanh), adding regularization
# # (via dropout) or use a smaller learning rate and/or momentum.
# ####
# #############################################3
# #When trying to use the lenet model on the line 200
# #I define in the lenet function below, H2O crashes:
# In the terminal running the h2o I get the crash message:
# [07:25:04] src/ndarray/ndarray.cc:756: Check failed: (dshape.Size()) == (size) Memory size do not match
# #
#
# And this is the full log of the execution of the model training in the terminal:
# 02-09 07:25:01.395 127.0.0.1:54321       91717  FJ-1-171  INFO: Dropping ignored columns: [Month1, Lightning, Precipitation, Deterioration, Min_Temp]
# 02-09 07:25:01.396 127.0.0.1:54321       91717  FJ-1-171  INFO: Rebalancing train dataset into 3 chunks.
# 02-09 07:25:01.698 127.0.0.1:54321       91717  FJ-1-171  INFO: _replicate_training_data: Disabling replicate_training_data on 1 node.
# 02-09 07:25:01.698 127.0.0.1:54321       91717  FJ-1-171  INFO: _problem_type: Automatically selecting problem_type: dataset
# 02-09 07:25:01.698 127.0.0.1:54321       91717  FJ-1-171  INFO: _categorical_encoding: Automatically enabling OneHotInternal categorical encoding.
# 02-09 07:25:01.699 127.0.0.1:54321       91717  FJ-1-171  INFO: _network_definition_file: Automatically setting network type to 'user', since a network definition file was provided.
# 02-09 07:25:01.742 127.0.0.1:54321       91717  FJ-1-171  INFO: Loading the network from: /tmp/symbol_lenet-py.json
# 02-09 07:25:01.743 127.0.0.1:54321       91717  FJ-1-171  INFO: Setting the optimizer and initializing the first and last layer.
# Loading H2O mxnet bindings.
# Found CUDA_PATH environment variable, trying to connect to GPU devices.
# Loading CUDA library.
# Loading mxnet library.
# Loading H2O mxnet bindings.
# Done loading H2O mxnet bindings.
# Constructing model.
# Done constructing model.
# Building network.
# Loading the model.
# Done loading the model.
# mxnet data input shape: (32,2)
# Setting the optimizer.
# Initializing state.
# Done creating the model.
# Done building network.
# 02-09 07:25:03.671 127.0.0.1:54321       91717  FJ-1-171  WARN: No network parameters file specified. Starting from scratch.
# 02-09 07:25:03.671 127.0.0.1:54321       91717  FJ-1-171  INFO: Native backend -> Java.
# Saving the model.
# Done saving the model.
# Saving the model parameters.
# Done saving the model parameters.
# 02-09 07:25:03.960 127.0.0.1:54321       91717  FJ-1-171  INFO: Took:  0.289 sec
# 02-09 07:25:03.961 127.0.0.1:54321       91717  FJ-1-171  INFO: Building the model on 2 numeric features and 0 (one-hot encoded) categorical features.
# 02-09 07:25:04.014 127.0.0.1:54321       91717  FJ-1-171  INFO: Model category: Regression
# 02-09 07:25:04.014 127.0.0.1:54321       91717  FJ-1-171  INFO: Approximate number of model parameters (weights/biases/aux): 1,310,331
# 02-09 07:25:04.020 127.0.0.1:54321       91717  FJ-1-171  INFO: One epoch corresponds to 2637 training data rows.
# 02-09 07:25:04.021 127.0.0.1:54321       91717  FJ-1-171  INFO: Number of chunks of the training data: 3
# 02-09 07:25:04.021 127.0.0.1:54321       91717  FJ-1-171  INFO: Number of chunks of the validation data: 1
# 02-09 07:25:04.026 127.0.0.1:54321       91717  FJ-1-171  INFO: Total setup time:  2.924 sec
# 02-09 07:25:04.026 127.0.0.1:54321       91717  FJ-1-171  INFO: Starting to train the Deep Learning model.
# 02-09 07:25:04.027 127.0.0.1:54321       91717  FJ-1-171  INFO: Automatically enabling data caching, expecting to require 10.7 KB.
# [07:25:04] /home/norayr/1MyDataBases-short/100deepwater-master/deepwater/thirdparty/mxnet/dmlc-core/include/dmlc/logging.h:235: [07:25:04] src/ndarray/ndarray.cc:756: Check failed: (dshape.Size()) == (size) Memory size do not match
# #
# # A fatal error has been detected by the Java Runtime Environment:
# #
# #  SIGSEGV (0xb) at pc=0x00007f8f7dd80cd0, pid=91717, tid=0x00007f8f957d7700
# #
# # JRE version: OpenJDK Runtime Environment (8.0_111-b15) (build 1.8.0_111-b15)
# # Java VM: OpenJDK 64-Bit Server VM (25.111-b15 mixed mode linux-amd64 )
# # Problematic frame:
# # C  [libmxnet.so+0x426cd0]  mxnet::CopyFromTo(mxnet::NDArray const&, mxnet::NDArray*, int)+0x20
# #
# # Failed to write core dump. Core dumps have been disabled. To enable core dumping, try "ulimit -c unlimited" before starting Java again
# #
# # An error report file with more information is saved as:
# # /home/norayrm/100deepwater-master/h2o-3/build/hs_err_pid91717.log
# #
# # If you would like to submit a bug report, please visit:
# #   http://bugreport.java.com/bugreport/crash.jsp
# # The crash happened outside the Java Virtual Machine in native code.
# # See problematic frame for where to report the bug.
# #
# Aborted (core dumped)
#
#Also see the log in the attached file hs_err_pid91717.log
#########################################################################33

import pandas as pd
import h2o
import os
from h2o import exceptions
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import plotly
import plotly.graph_objs as go
from datetime import datetime

from h2o.estimators.deepwater import H2ODeepWaterEstimator

import mxnet as mx

# Start h2o
#h2o.init(nthreads=71, max_mem_size='30G')
h2o.init(ip="localhost",strict_version_check=False)

# Remove all objects from h2o
# h2o.remove_all()

# Import data to pandas dataframe
data_full = pd.read_csv(
            os.path.join(os.environ.get('HOME'), '0MyDataBases/40Python/URD_DNN/data/ExportFileR.csv'))

# Set start and end dates for training
date_start = '2006-Jan-01'
date_end = '2014-Dec-31'

# Find row indices of training data
start_row = data_full[data_full['Date1'] == date_start].index.tolist()[0]
end_row = data_full[data_full['Date1'] == date_end].index.tolist()[0]

# Create training data slice and convert to H2OFrame
train_pd = data_full[start_row:end_row+1].copy()
train_pd.drop(['Date1'], axis=1, inplace=True)

print(train_pd.head())

train = h2o.H2OFrame(train_pd, column_types=['int', 'enum', 'real', 'real', 'int','int', 'int', 'int'],destination_frame = 'train_prep.h2o')
training, validation = train.split_frame(ratios=[0.8], destination_frames = ['training.h2o','validation.h2o'])

# Create test data slice and convert to H2OFrame
test_pd = data_full[end_row + 1:].copy()
test_pd.drop('Date1', axis=1, inplace=True)
test = h2o.H2OFrame(test_pd,   column_types=['int', 'enum', 'real', 'real', 'int', 'int', 'int', 'int'],destination_frame = 'test.h2o')

# Define predictors and output
predictors = list(train.columns)[1:]
output = list(train.columns)[0]

print(predictors)
##############################
## begin defining lenet #####

def lenet(num_classes):
    import mxnet as mx
    #Define the input data
    data = mx.symbol.Variable('data')

    # A fully connected hidden layer
    # data: input source
    # num_hidden: number of neurons in this hidden layer

    fc1 = mx.symbol.FullyConnected(data, num_hidden=num_classes)
    tanh1 = mx.symbol.Activation(data=fc1, act_type="tanh") 

    # second fullc 

    fc2 = mx.symbol.FullyConnected(data=tanh1, num_hidden=num_classes)
    tanh2 = mx.symbol.Activation(data=fc2, act_type="tanh")
    # third fullc
    fc3 = mx.symbol.FullyConnected(data=tanh2, num_hidden=num_classes)

    # Use linear regression for the output layer

    lenet= mx.symbol.LinearRegressionOutput(fc3)

    # loss
    # lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return lenet;

## end lenet definiton
#################################################################

num_classes=800

mxnet_model = lenet(num_classes)
# To import the model inside the DeepWater training engine we need to save the model to a file:

model_filename="/tmp/symbol_lenet-py.json"

mxnet_model.save(model_filename)

# pip install graphviz
# sudo apt-get install graphviz

import graphviz
mx.viz.plot_network(mxnet_model, shape={"data":(1, 1, 28, 28)}, node_attrs={"shape":'rect',"fixedsize":'false'})

###############

# lenet_model.train(x=train_df.names, y=y, training_frame=train_df, validation_frame=test_df)


# Run custom  lenet  DNN
model_id = 'Python_-lenet-URD_DNN_2006-2014' 
# model = H2ODeepLearningEstimator(model_id=model_id, epochs=500, hidden=[800,800], activation ="Tanh", l1=0, l2=0,stopping_rounds=5)


print('about to train ', model_id)

#### The model using mxnet defining the model in the json file
model = H2ODeepWaterEstimator(model_id=model_id,  network_definition_file=model_filename,  epochs=500,stopping_rounds=0)

##### The feedforward DeepWater model
# model = H2ODeepWaterEstimator(model_id=model_id, hidden=[80,80], activation ="Tanh",  epochs=500, stopping_rounds=5)

print("predictors are ",predictors)

### trying to use differenc combinations of the predictors still fails with the exponential eror
pred_rest=[predictors[1],predictors[5]]

print(pred_rest)

### Trying different combinations of the predictors
model.train(x=predictors, y=output, training_frame=training, validation_frame=validation)
#model.train(x=pred_rest, y=output, training_frame=training, validation_frame=validation)


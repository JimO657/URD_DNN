### Lenet mxnet version
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
h2o.init(ip="192.168.0.21",strict_version_check=False)

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

#!head -n 20 $model_filename


#############################
## lenet model

# lenet_model = H2ODeepWaterEstimator(
#     epochs=10,
#     learning_rate=1e-3, 
#     mini_batch_size=64,
#     network_definition_file=model_filename,
# #    network='lenet',            ## equivalent pre-configured model
# #    image_shape=[28,28],
#     problem_type='dataset',      ## Not 'image' since we're not passing paths to image files, but raw numbers
# #    ignore_const_cols=False,     ## We need to keep all 28x28=784 pixel values, even if some are always 0
# #    channels=1
# )

###############

# lenet_model.train(x=train_df.names, y=y, training_frame=train_df, validation_frame=test_df)


# Run custom  lenet  DNN
model_id = 'Python_-lenet-URD_DNN_2006-2014' 
# #model = H2ODeepLearningEstimator(model_id=model_id, epochs=500, hidden=[800,800,800], activation ="Tanh", l1=0, l2=0,stopping_rounds=0)
# model = H2ODeepLearningEstimator(model_id=model_id, epochs=500, hidden=[80], activation ="tanh",stopping_rounds=0)

print('about to train ', model_id)

# model = H2ODeepWaterEstimator(model_id=model_id,  network_definition_file=model_filename,  epochs=500,stopping_rounds=0)


model = H2ODeepWaterEstimator(model_id=model_id, hidden=[80,80], activation ="Tanh",  epochs=500, stopping_rounds=0)
print("predictors are ",predictors)
pred_rest=predictors[2]
print(pred_rest)
model.train(x=pred_rest, y=output, training_frame=training, validation_frame=validation)


# Define list of pandas DataFrames for model to predict on
base_data_path = 'C:\\0MyDataBases\\40Python\URD_DNN\data_2015'
# l_csv_test_data = ['ExportFileWeather_2015.csv', 'ExportFileWeather_2014.csv', 'ExportFileWeather_2013.csv',
#                    'ExportFileWeather_2012.csv', 'ExportFileWeather_2011.csv', 'ExportFileWeather_2010.csv']

l_csv_test_data = ['ExportFileWeather_2010.csv']

l_pd_test_data = [data_full]
for csv_test_data in l_csv_test_data:
    l_pd_test_data.append(pd.read_csv(os.path.join(base_data_path, csv_test_data)))

# Get model predictions on test data
l_predictions_raw = get_predictions(model, l_pd_test_data)

# Add prediction column to existing pandas DataFrames
for i in range(len(l_predictions_raw)):
    l_pd_test_data[i]['Prediction'] = l_predictions_raw[i]['predict']

# Aggregate full data into yearly, monthly, and daily results with errors
l_predictions = []
for pd_test_data in tqdm(l_pd_test_data):
    l_predictions.append(aggregate_by_day_month_year(pd_test_data))

# Create list of strings indicating year of test data being used
l_test_year = []# ['2016']
for filename in l_csv_test_data:
    l_test_year.append(filename[-8:-4])

# Zip weather data year with predictions and errors
d_predictions = {}
for i in range(len(l_test_year)):
    d_predictions[l_test_year[i]] = l_predictions[i]

# Create traces for predictions
d_traces = {}
for weather_year in d_predictions:
    for time_frame in d_predictions[weather_year]:
        try:
            d_traces[weather_year][time_frame] = {}
            d_traces[weather_year][time_frame]['Prediction'] = go.Scatter(
                x=d_predictions[weather_year][time_frame]['Date'],
                y=d_predictions[weather_year][time_frame]['Prediction'],
                name=time_frame + ' Prediction ' + weather_year,
                line=dict(dash='dash'),
                legendgroup=time_frame
            )
            d_traces[weather_year][time_frame]['Error'] = go.Bar(
                x=d_predictions[weather_year][time_frame]['Date'],
                y=d_predictions[weather_year][time_frame]['Error'],
                name=time_frame + ' Error ' + weather_year,
                legendgroup=time_frame
            )

        except KeyError:
            d_traces[weather_year] = {}
            d_traces[weather_year][time_frame] = {}
            d_traces[weather_year][time_frame]['Prediction'] = go.Scatter(
                x=d_predictions[weather_year][time_frame]['Date'],
                y=d_predictions[weather_year][time_frame]['Prediction'],
                name=time_frame + ' Prediction ' + weather_year,
                line=dict(dash='dash'),
                legendgroup=time_frame
            )
            d_traces[weather_year][time_frame]['Error'] = go.Bar(
                x=d_predictions[weather_year][time_frame]['Date'],
                y=d_predictions[weather_year][time_frame]['Error'],
                name=time_frame + ' Error ' + weather_year,
                legendgroup=time_frame
            )

# Create data and visibility dictionaries
l_vis_dicts = []
for test_year in l_test_year:
    visibility = [True, True, True]
    l_data = [trace_yearly]
    for weather_year in d_traces:
        for time_frame in d_traces[weather_year]:
            for data_type in d_traces[weather_year][time_frame]:
                l_data.append(d_traces[weather_year][time_frame][data_type])
                if weather_year == test_year:
                    visibility.append(True)
                else:
                    visibility.append(False)
    l_vis_dicts.append({'args': ['visible', visibility],
                        'label': test_year,
                        'method': 'restyle'
                        })

    data = go.Data(l_data)

# Create layout for plotly
layout = go.Layout(
    title='URD Prediction vs. Actual',
    xaxis=dict(title='', rangeslider=dict(thickness=0.1), type='date', showgrid=True),
    yaxis=dict(title='', showgrid=True),
    updatemenus=list([
        dict(
            buttons=[vis_dict for vis_dict in l_vis_dicts],
            type='buttons'
        )]))

# Plot with plotly
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig)


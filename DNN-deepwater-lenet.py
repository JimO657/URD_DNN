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
h2o.init(strict_version_check=False)

# Remove all objects from h2o
h2o.remove_all()

# Import data to pandas dataframe
data_full = pd.read_csv(
            os.path.join(os.environ.get('HOME'), '0MyDataBases/7R/ADHOC_Qlikview-linux/data/ExportFileR.csv'))

# Set start and end dates for training
date_start = '2006-Jan-01'
date_end = '2014-Dec-31'

# Find row indices of training data
start_row = data_full[data_full['Date1'] == date_start].index.tolist()[0]
end_row = data_full[data_full['Date1'] == date_end].index.tolist()[0]

# Create training data slice and convert to H2OFrame
train_pd = data_full[start_row:end_row+1].copy()
train_pd.drop(['Date1','Lightning','Snow','Precipitation'], axis=1, inplace=True)

#train_pd.


print 'hi'
print(train_pd.head())

train = h2o.H2OFrame(train_pd, column_types=['int', 'enum', 'real', 'real', 'int'],destination_frame = 'train.h2o')
training, validation = train.split_frame(ratios=[0.8], destination_frames = ['training.h2o','validation.h2o'])

# Create test data slice and convert to H2OFrame
test_pd = data_full[end_row + 1:].copy()
test_pd.drop('Date1', axis=1, inplace=True)
test = h2o.H2OFrame(test_pd, column_types=['int', 'enum', 'real', 'real', 'int', 'int', 'int', 'int'],destination_frame = 'test.h2o')

# Define predictors and output
predictors = list(train.columns)[2:]
output = list(train.columns)[0]

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

print'about to train '

# model = H2ODeepWaterEstimator(model_id=model_id,  network_definition_file=model_filename,  epochs=500,stopping_rounds=0)


model = H2ODeepWaterEstimator(model_id=model_id, hidden=[800,800,800], activation ="Tanh",  epochs=500,stopping_rounds=0)


model.train(x=predictors, y=output, training_frame=training, validation_frame=validation)

# # Save DNN model
# save_path = os.path.join(os.environ.get('HOME'), '0MyDataBases/7R/ADHOC_Qlikview-linux/H2O_Models/')
# try:
#     h2o.save_model(model, path=save_path)
# except exceptions.H2OServerError:
#     os.remove(save_path + model_id)
#     h2o.save_model(model, path=save_path)

# # Run model prediction on original data
# original_prediction = model.predict(test)

# # Import weather data to pandas dataframe
# data_weather = pd.read_csv(
#                os.path.join(
#                os.environ.get('HOME'), '0MyDataBases/7R/ADHOC_Qlikview-linux/data/ExportFileWeather_2010.csv'))
# data_weather_nodate = data_weather.drop('Date1', 1)
# test_weather = h2o.H2OFrame(data_weather_nodate,
#                             column_types=['int', 'enum', 'real', 'real', 'int', 'int', 'int', 'int'],destination_frame = 'test-weather.h2o')
# weather_prediction = model.predict(test_weather)

# # Add Year+Month column to full data
# times = pd.DatetimeIndex(data_full.Date1)
# pd_real_bymonth = data_full.groupby([times.year, times.month]).sum()
# pd_real_bymonth.reset_index(inplace=True)
# pd_real_bymonth = pd_real_bymonth.rename(columns={'level_0': 'Year', 'level_1': 'Month'})
# pd_real_bymonth['Date'] = pd_real_bymonth.apply(lambda row: datetime(int(row['Year']), int(row['Month']), 1), axis=1)

# # Add Year+Month column to weather data
# times2 = pd.DatetimeIndex(data_weather.Date1)
# pd_pred_bymonth = data_weather.groupby([times2.year, times2.month]).sum()
# pd_pred_bymonth.reset_index(inplace=True)
# pd_pred_bymonth = pd_pred_bymonth.rename(columns={'level_0': 'Year', 'level_1': 'Month'})
# pd_pred_bymonth['Date'] = pd_pred_bymonth.apply(lambda row: datetime(int(row['Year']), int(row['Month']), 1), axis=1)

# # Add Year+Month column to prediction
# pd_weather_prediction = weather_prediction.as_data_frame()

# times3 = pd.DatetimeIndex(data_full.Date1)
# pd_predict = pd_weather_prediction.groupby([times3.year, times3.month]).sum()
# pd_predict.reset_index(inplace=True)
# pd_predict= pd_predict.rename(columns={'level_0': 'Year', 'level_1': 'Month'})
# pd_predict['Date'] = pd_predict.apply(lambda row: datetime(int(row['Year']), int(row['Month']), 1), axis=1)

# # Create dataframe for error between prediction and real data
# error = pd_predict.predict - pd_real_bymonth.ACT

# # Plot with plotly
# trace1 = go.Scatter(x=pd_real_bymonth['Date'], y=pd_real_bymonth['ACT'], name='Real')
# trace2 = go.Scatter(x=pd_predict['Date'], y=pd_predict['predict'], name='Predicted')
# trace3 = go.Bar(x=pd_real_bymonth['Date'], y=error, name='Error')
# data = [trace1, trace2, trace3]
# layout = dict(title='URD Prediction vs. Actual',
#               xaxis=dict(title='Date', rangeslider=dict(), type='date'),
#               yaxis=dict(title='ACT'),
#               )
# fig = dict(data=data, layout=layout)
# plotly.offline.plot(fig)

# Define list of pandas DataFrames for model to predict on
base_data_path = 'C:\\0MyDataBases\\7R\ADHOC_Qlikview-linux\data_2015'
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

# Create traces for actual data
# trace_yearly = go.Scatter(
#     x=d_predictions['2016']['Yearly']['Date'],
#     y=d_predictions['2016']['Yearly']['ACT'],
#     name='Yearly Actual',
#     legendgroup='Yearly',
# )
# trace_monthly = go.Scatter(
#     x=d_predictions['2016']['Monthly']['Date'],
#     y=d_predictions['2016']['Monthly']['ACT'],
#     name='Monthly Actual',
#     legendgroup='Monthly',
# )
# trace_daily = go.Scatter(
#     x=d_predictions['2016']['Daily']['Date'],
#     y=d_predictions['2016']['Daily']['ACT'],
#     name='Daily Actual',
#     legendgroup='Daily',
# )

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


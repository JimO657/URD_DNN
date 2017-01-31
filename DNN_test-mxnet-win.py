
import pandas as pd
import h2o
import os
from h2o import exceptions
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import plotly
import plotly.graph_objs as go
from datetime import datetime
import tqdm

#test mxnet
import mxnet as mx;
#a = mx.nd.ones((2, 3));
#print ((a*2).asnumpy());


# Start h2o
h2o.init(nthreads=3, max_mem_size='10G')
#h2o.init(strict_version_check=False)

# Remove all objects from h2o
h2o.remove_all()


# Import data to pandas dataframe
infpath='C:\\0MyDataBases\\7R\ADHOC_Qlikview-linux\data_2015\ExportFileR.csv'

data_full = pd.read_csv(infpath)

# Set start and end dates for training
date_start = '2006-Jan-01'
date_end = '2014-Dec-31'

# Find row indices of training data
start_row = data_full[data_full['Date1'] == date_start].index.tolist()[0]
end_row = data_full[data_full['Date1'] == date_end].index.tolist()[0]

# Create training data slice and convert to H2OFrame
train_pd = data_full[start_row:end_row+1].copy()
train_pd.drop('Date1', axis=1, inplace=True)
train = h2o.H2OFrame(train_pd, column_types=['int', 'enum', 'real', 'real', 'int', 'int', 'int', 'int'])
training, validation = train.split_frame(ratios=[0.8], destination_frames = ['train.h2o','valid.h2o'])

# Create test data slice and convert to H2OFrame
test_pd = data_full[end_row + 1:].copy()
test_pd.drop('Date1', axis=1, inplace=True)
test = h2o.H2OFrame(test_pd, column_types=['int', 'enum', 'real', 'real', 'int', 'int', 'int', 'int'],destination_frame = 'test.h2o')

# Define predictors and output
predictors = list(train.columns)[2:]
output = list(train.columns)[0]

####### Starting Lenet part

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

#model_filename="/tmp/symbol_lenet-py.json"

#mxnet_model.save(model_filename)

########################################
### Training mxnet







# Run DNN
model_id = 'Python_URD_DNN_2006-2014'
model = H2ODeepLearningEstimator(model_id=model_id, epochs=5, hidden=[800,800], activation ="Tanh", l1=0, l2=0,stopping_rounds=5,stopping_metric= 'MSE',stopping_tolerance=1e-6)

model.train(x=predictors, y=output, training_frame=training, validation_frame=validation)







def get_predictions(urd_model, l_test_data):
    """Creates predictions on test data using H2O model

    Args:
        urd_model (H2ODeepLearningEstimator): H2O model for prediction.
        l_test_data (list): list of pandas DataFrames to predict on.

    Returns:
        list: List of pandas DataFrames containing predictions.

    """

    # Initialize list of predictions
    predictions = []

    # Convert test data to H2OFrames and run predictions
    for test_data in l_test_data:
        test_data_no_date = test_data.drop('Date1', axis=1, inplace=False)
        h2o_test = h2o.H2OFrame(test_data_no_date,
                                column_types=['int', 'enum', 'real', 'real', 'int', 'int', 'int', 'int'])
        predictions.append(urd_model.predict(h2o_test).as_data_frame())

    return predictions


def aggregate_by_day_month_year(dataframe):
    """Aggregates pandas DataFrames by day, month, and year using indices

    Args:
        dataframe (pandas DataFrame): DataFrame with 'Date1' column.

    Returns:
        dictionary: Maps words 'Daily', 'Monthly', and 'Yearly' to aggregated pandas DataFrame

    """

    # Create time index
    times = pd.DatetimeIndex(dataframe.Date1)

    # Create daily aggregate
    pd_daily = dataframe.groupby([times.year, times.month, times.day]).sum() # Aggregate by day
    pd_daily.reset_index(inplace=True)  # Turns multi index into columns
    pd_daily = pd_daily.rename(columns={'level_0': 'Year', 'level_1': 'Month', 'level_2': 'Day'})
    pd_daily['Date'] = pd_daily.apply(lambda row: datetime(int(row['Year']), int(row['Month']), int(row['Day']), 1), axis=1)

    # Create monthly aggregate
    pd_monthly = dataframe.groupby([times.year, times.month]).sum() # Aggregate by month
    pd_monthly.reset_index(inplace=True)  # Turns multi index into columns
    pd_monthly = pd_monthly.rename(columns={'level_0': 'Year', 'level_1': 'Month'})  # Rename index columns
    pd_monthly['Date'] = pd_monthly.apply(lambda row: datetime(int(row['Year']), int(row['Month']), 1), axis=1)

    # Create yearly aggregate
    pd_yearly = dataframe.groupby([times.year]).sum()
    pd_yearly.reset_index(inplace=True)
    pd_yearly = pd_yearly.rename(columns={'index': 'Date'})

    # Create error columns
    pd_daily['Error'] = pd_daily['Prediction'] - pd_daily['ACT']
    pd_monthly['Error'] = pd_monthly['Prediction'] - pd_monthly['ACT']
    pd_yearly['Error'] = pd_yearly['Prediction'] - pd_yearly['ACT']

    return {'Daily': pd_daily, 'Monthly': pd_monthly, 'Yearly': pd_yearly}


# Save DNN model
save_path = "C:\\0MyDataBases\\7R\ADHOC_Qlikview-linux\H2O_Models_2015\\"
try:
    h2o.save_model(model, path=save_path)
except exceptions.H2OServerError:
    os.remove(save_path + model_id)
    h2o.save_model(model, path=save_path)

# Run model prediction on original data
original_prediction = model.predict(test)

# Import weather data to pandas dataframe
data_weather = pd.read_csv('C:\\0MyDataBases\\7R\ADHOC_Qlikview-linux\data_2015\ExportFileWeather_2010.csv')
data_weather_nodate = data_weather.drop('Date1', 1)


test_weather = h2o.H2OFrame(data_weather_nodate, column_types=['int', 'enum', 'real', 'real', 'int', 'int', 'int', 'int'])
weather_prediction = model.predict(test_weather)

# Import weather data to pandas dataframe
data_weather = pd.read_csv('C:\\0MyDataBases\\7R\ADHOC_Qlikview-linux\data_2015\ExportFileWeather_2010.csv')
data_weather_nodate = data_weather.drop('Date1', 1)


test_weather = h2o.H2OFrame(data_weather_nodate, column_types=['int', 'enum', 'real', 'real', 'int', 'int', 'int', 'int'])
weather_prediction = model.predict(test_weather)

# Define list of pandas DataFrames for model to predict on
base_data_path = 'C:\\0MyDataBases\\7R\ADHOC_Qlikview-linux\data_2015'
l_csv_test_data = ['ExportFileWeather_2015.csv', 'ExportFileWeather_2014.csv', 'ExportFileWeather_2013.csv',
                   'ExportFileWeather_2012.csv', 'ExportFileWeather_2011.csv', 'ExportFileWeather_2010.csv']
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
l_test_year = ['2016']
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
trace_yearly = go.Scatter(
    x=d_predictions['2016']['Yearly']['Date'],
    y=d_predictions['2016']['Yearly']['ACT'],
    name='Yearly Actual',
    legendgroup='Yearly',
)
trace_monthly = go.Scatter(
    x=d_predictions['2016']['Monthly']['Date'],
    y=d_predictions['2016']['Monthly']['ACT'],
    name='Monthly Actual',
    legendgroup='Monthly',
)
trace_daily = go.Scatter(
    x=d_predictions['2016']['Daily']['Date'],
    y=d_predictions['2016']['Daily']['ACT'],
    name='Daily Actual',
    legendgroup='Daily',
)

# Create data and visibility dictionaries
l_vis_dicts = []
for test_year in l_test_year:
    visibility = [True, True, True]
    l_data = [trace_yearly, trace_monthly, trace_daily]
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


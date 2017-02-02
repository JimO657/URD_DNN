import pandas as pd
import h2o
import os
from h2o import exceptions
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import plotly
import plotly.graph_objs as go
from datetime import datetime


##### Start h2o
#h2o.init(nthreads=71, max_mem_size='30G')
h2o.init(strict_version_check=False)

# Remove all objects from h2o
h2o.remove_all()

# Import data to pandas dataframes
data_full = pd.read_csv(
            os.path.join(
            os.environ.get('HOME'), '0MyDataBases/7R/ADHOC_Qlikview-linux/data_2015/ExportFileR.csv'))

# Set start and end dates for training
date_start = '2006-Jan-01'
date_end = '2014-Dec-31'

# Find row indices of training data
start_row = data_full[data_full['Date1'] == date_start].index.tolist()[0]
end_row = data_full[data_full['Date1'] == date_end].index.tolist()[0]

# Create training data slice and convert to training and validation H2OFrames
data_full_pd_train = data_full[start_row:end_row+1].copy()
data_full_pd_train_nodate = data_full_pd_train.drop('Date1', axis=1, inplace=False)
train = h2o.H2OFrame(data_full_pd_train_nodate,
                     column_types=['int', 'enum', 'real', 'real', 'int', 'int', 'int', 'int'])
training, validation = train.split_frame(ratios=[0.8])

# Create test data slice and convert to H2OFrame
test_pd = data_full[end_row + 1:].copy()
test_pd_nodate = test_pd.drop('Date1', axis=1, inplace=False)
test = h2o.H2OFrame(test_pd_nodate, column_types=['int', 'enum', 'real', 'real', 'int', 'int', 'int', 'int'])

# Define predictors and response
predictors = list(train.columns)[2:]
response = list(train.columns)[0]

# Run DNN
model_id = 'Python_URD_DNN_2006-2014'
model = H2ODeepLearningEstimator(model_id=model_id, epochs=5000, hidden=[800,800], activation ="Tanh",
                                 l1=0, l2=0,stopping_rounds=5,stopping_metric= 'MSE',stopping_tolerance=1e-6)
model.train(x=predictors, y=response, training_frame=training, validation_frame=validation)

# Save DNN model
save_path = os.path.join(os.environ.get('HOME'), '0MyDataBases/7R/ADHOC_Qlikview-linux/H2O_Models/')
try:
    h2o.save_model(model, path=save_path)
except exceptions.H2OServerError:  # Raised if file already exists
    os.remove(save_path + model_id)
    h2o.save_model(model, path=save_path)

# Run model prediction on original data
prediction_original = model.predict(test)

# Run model prediction on 2010 data
data_weather2010 = pd.read_csv(
               os.path.join(
               os.environ.get('HOME'), '0MyDataBases/7R/ADHOC_Qlikview-linux/data_2015/ExportFileWeather_2010.csv'))
data_weather2010_nodate = data_weather2010.drop('Date1', 1)
test_weather2010 = h2o.H2OFrame(data_weather2010_nodate,
                            column_types=['int', 'enum', 'real', 'real', 'int', 'int', 'int', 'int'])

prediction_weather2010 = model.predict(test_weather2010)

# Group full data by day, month, and year
times = pd.DatetimeIndex(data_full.Date1)

pd_real_byday = data_full.rename(columns={'Date1': 'Date'})
pd_real_bymonth = pd_real_byday.groupby([times.year, times.month]).sum()
pd_real_byyear = pd_real_byday.groupby([times.year]).sum()

pd_real_bymonth.reset_index(inplace=True)  # Turns multi index into columns
pd_real_bymonth = pd_real_bymonth.rename(columns={'level_0': 'Year', 'level_1': 'Month'})  # Rename index columns
pd_real_bymonth['YearMonth'] = pd_real_bymonth.apply(
            lambda row: datetime(int(row['Year']), int(row['Month']), 1), axis=1)  # Create column for year and month

pd_real_byyear.reset_index(inplace=True)
pd_real_byyear = pd_real_byyear.rename(columns={'index': 'Year'})

# Group original prediction by day, month, and year
times = pd.DatetimeIndex(test_pd.Date1)

pd_predreal_byday = prediction_original.as_data_frame()
pd_predreal_bymonth = pd_predreal_byday.groupby([times.year, times.month]).sum()
pd_predreal_byyear = pd_predreal_byday.groupby([times.year]).sum()

pd_predreal_bymonth.reset_index(inplace=True)
pd_predreal_bymonth = pd_predreal_bymonth.rename(columns={'level_0': 'Year', 'level_1': 'Month'})
pd_predreal_bymonth['YearMonth'] = pd_predreal_bymonth.apply(
            lambda row: datetime(int(row['Year']), int(row['Month']), 1), axis=1)

pd_predreal_byyear.reset_index(inplace=True)
pd_predreal_byyear = pd_predreal_byyear.rename(columns={'index': 'Year'})

# Group 2010 weather prediction by day, month, and year
times = pd.DatetimeIndex(data_weather2010.Date1)

pd_pred2010_byday = prediction_weather2010.as_data_frame()
pd_pred2010_bymonth = pd_pred2010_byday.groupby([times.year, times.month]).sum()
pd_pred2010_byyear = pd_pred2010_byday.groupby([times.year]).sum()

pd_pred2010_bymonth.reset_index(inplace=True)
pd_pred2010_bymonth = pd_pred2010_bymonth.rename(columns={'level_0': 'Year', 'level_1': 'Month'})
pd_pred2010_bymonth['YearMonth'] = pd_pred2010_bymonth.apply(
            lambda row: datetime(int(row['Year']), int(row['Month']), 1), axis=1)

pd_pred2010_byyear.reset_index(inplace=True)
pd_pred2010_byyear = pd_pred2010_byyear.rename(columns={'index': 'Year'})

# Create dataframes for error between prediction and real data
error_real_byday = pd_predreal_byday.predict - pd_real_byday.ACT
error_real_bymonth = pd_predreal_bymonth.predict - pd_real_bymonth.ACT
error_real_byyear = pd_predreal_byyear.predict - pd_real_byyear.ACT

error_2010_byday = pd_pred2010_byday.predict - pd_real_byday.ACT
error_2010_bymonth = pd_pred2010_bymonth.predict - pd_real_bymonth.ACT
error_2010_byyear = pd_pred2010_byyear.predict - pd_real_byyear.ACT

# Plot with plotly
trace_real_byday = go.Scatter(x=pd_real_byday['Date'], y=pd_real_byday['ACT'])
trace_real_bymonth = go.Scatter(x=pd_real_bymonth['Date'], y=pd_real_bymonth['ACT'])
trace_real_byyear = go.Scatter(x=pd_real_byyear['Year'], y=pd_real_byyear['ACT'])

trace_predreal_byday = go.Scatter(x=pd_predreal_byday['Date'], )

trace1 = go.Scatter(x=pd_real_bymonth['YearMonth'], y=pd_real_bymonth['ACT'], name='Real')
trace2 = go.Scatter(x=pd_predict['Date'], y=pd_predict['predict'], name='Predicted')
trace3 = go.Bar(x=pd_real_bymonth['Date'], y=error, name='Error')
data = [trace1, trace2, trace3]

layout = dict(title='URD Prediction vs. Actual',
              xaxis=dict(title='Date', rangeslider=dict(), type='date'),
              yaxis=dict(title='ACT'),
              )

fig = dict(data=data, layout=layout)

plotly.offline.plot(fig)

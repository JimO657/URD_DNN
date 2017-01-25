import pandas as pd
import h2o
import os
from h2o import exceptions
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.deepwater import H2ODeepWaterEstimator


if __name__=='__main__':

    # Start h2o
    h2o.init(nthreads=71, max_mem_size='30G')

    # Remove all objects from h2o
    h2o.remove_all()

    # Import data to pandas dataframe
    data_full = pd.read_csv(
        '/home/norayr/1MyDataBases-short/100deepwater-master/ADHOC_Qlikview/data_2015/ExportFileR.csv')

    # Set start and end dates for training
    date_start = '2006-Jan-01'
    date_end = '2014-Dec-31'

    # Find row indices of training data
    start_row = data_full[data_full['Date1'] == date_start].index.tolist()[0]
    end_row = data_full[data_full['Date1'] == date_end].index.tolist()[0]

    # Create training data slice and convert to H2OFrame
    train_pd = data_full[start_row:end_row+1]
    train_pd.drop('Date1', axis=1, inplace=True)
    train = h2o.H2OFrame(train_pd, column_types=['int', 'enum', 'real', 'real', 'int', 'int', 'int', 'int'])
    training, validation = train.split_frame(ratios=[0.8])

    # Create test data slice and convert to H2OFrame
    test_pd = data_full[end_row + 1:]#6844]
    test_pd.drop('Date1', axis=1, inplace=True)
    test = h2o.H2OFrame(test_pd, column_types=['int', 'enum', 'real', 'real', 'int', 'int', 'int', 'int'])

    # Define predictors and output
    predictors = list(train.columns)[2:]
    output = list(train.columns)[0]

    # Run DNN
    model_id = 'Python_URD_Deepwater_2006-2014'
    model = H2ODeepWaterEstimator(model_id=model_id, epochs=50, hidden=[800,800], activation="Tanh",
                                     l1=0, l2=0, score_training_samples=5, score_validation_samples=5)
    model.train(x=predictors, y=output, training_frame=training, validation_frame=validation)

    # Save DNN model
    save_path = '/home/norayr/1MyDataBases-short/100deepwater-master/ADHOC_Qlikview/'
    try:
        h2o.save_model(model, path=save_path)
    except exceptions.H2OServerError:
        os.remove(save_path)
        h2o.save_model(model, path=save_path)

    # Run model prediction on original data
    original_prediction = model.predict(test)

    # Import weather data to pandas dataframe
    data_weather = pd.read_csv(
        '/home/norayr/1MyDataBases-short/100deepwater-master/ADHOC_Qlikview/data_2015/ExportFileWeather_2010.csv')
    data_weather.drop('Date1', axis=1, inplace=True)
    test_weather = h2o.H2OFrame(data_weather, column_types=['int', 'enum', 'real', 'real', 'int', 'int', 'int', 'int'])
    weather_prediction = model.predict(test_weather)
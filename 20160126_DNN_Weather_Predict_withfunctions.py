import pandas as pd
import h2o
import os
from h2o import exceptions
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import plotly
import plotly.graph_objs as go
from datetime import datetime


def create_h2o_urd_model(urd_data):
    """Creates an H2O model from URD data

    Args:
        urd_data (pandas DataFrame): URD data as DataFrame.

    Returns:
        H2ODeepLearningEstimator: URD model.

    """

    # Start H2O and remove all objects
    h2o.init(strict_version_check=False)
    h2o.remove_all()

    # Set start and end dates for training
    date_start = '2006-Jan-01'
    date_end = '2014-Dec-31'

    # Find row indices of training data
    start_row = urd_data[urd_data['Date1'] == date_start].index.tolist()[0]
    end_row = urd_data[urd_data['Date1'] == date_end].index.tolist()[0]

    # Create training data slice and convert to training and validation H2OFrames
    urd_data_pd_train = urd_data[start_row:end_row + 1].copy()
    urd_data_pd_train_nodate = urd_data_pd_train.drop('Date1', axis=1, inplace=False)
    train = h2o.H2OFrame(urd_data_pd_train_nodate,
                         column_types=['int', 'enum', 'real', 'real', 'int', 'int', 'int', 'int'])
    training, validation = train.split_frame(ratios=[0.8])

    # Define predictors and response
    predictors = list(train.columns)[2:]
    response = list(train.columns)[0]

    # Run DNN
    urd_model_id = 'Python_URD_DNN_2006-2014'
    urd_model = H2ODeepLearningEstimator(model_id=urd_model_id, epochs=5000, hidden=[800, 800], activation="Tanh",
                                     l1=0, l2=0, stopping_rounds=5, stopping_metric='MSE', stopping_tolerance=1e-6)
    urd_model.train(x=predictors, y=response, training_frame=training, validation_frame=validation)

    # Save DNN model
    save_path = os.path.join(os.environ.get('HOME'), '0MyDataBases/7R/ADHOC_Qlikview-linux/H2O_Models/')
    try:
        h2o.save_model(urd_model, path=save_path)
    except exceptions.H2OServerError:  # Raised if file already exists
        os.remove(save_path + urd_model_id)
        h2o.save_model(urd_model, path=save_path)

    return urd_model


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
    pd_daily = data_full.rename(columns={'Date1': 'Date'})

    # Create monthly aggregate
    pd_monthly = pd_daily.groupby([times.year, times.month]).sum() # Aggregate by month
    pd_monthly.reset_index(inplace=True)  # Turns multi index into columns
    pd_monthly = pd_monthly.rename(columns={'level_0': 'Year', 'level_1': 'Month'})  # Rename index columns
    pd_monthly['Date'] = pd_monthly.apply(lambda row: datetime(int(row['Year']), int(row['Month']), 1), axis=1)

    # Create yearly aggregate
    pd_yearly = pd_daily.groupby([times.year]).sum()
    pd_yearly.reset_index(inplace=True)
    pd_yearly = pd_yearly.rename(columns={'index': 'Date'})

    return {'Daily': pd_daily, 'Monthly': pd_monthly, 'Yearly': pd_yearly}


if __name__ == "__main__":

    # Define home directory
    home_path = os.environ.get("HOME")

    # Import URD data
    urd_path = os.path.join(home_path, '0MyDataBases/7R/ADHOC_Qlikview-linux/data_2015/ExportFileR.csv')
    data_full = pd.read_csv(urd_path)

    # Create H2O model
    model = create_h2o_urd_model(data_full)

    # Define list of pandas DataFrames for model to predict on
    base_data_path = os.path.join(home_path, '0MyDataBases/7R/ADHOC_Qlikview-linux/data_2015')
    l_csv_test_data = ['ExportFileWeather_2010.csv']
    l_pd_test_data = [data_full]
    for csv_test_data in l_csv_test_data:
        l_pd_test_data.append(pd.read_csv(os.path.join(base_data_path, csv_test_data)))

    # Get model predictions on test data
    l_predictions_raw = get_predictions(model, l_pd_test_data)

    # Add prediction column to existing pandas DataFrames
    for i in range(len(l_predictions_raw)):
        l_pd_test_data[i]['Prediction'] = l_predictions_raw[i]['predict']

    # Create list of strings indicating year of test data being used
    l_test_year = ['2016']
    for filename in l_csv_test_data:
        l_test_year.append(filename[-8:-4])

    # Aggregate full data into yearly, monthly, and daily results
    d_predictions = []
    for pd_test_data in l_pd_test_data:
        d_predictions.append(aggregate_by_day_month_year(pd_test_data))

    # Get prediction errors

    # Zip weather data year with predictions and errors

import pandas as pd
import os
import platform

from DNN_plotly_functions_v001 import visualize_urd
from DNN_h2o_functions_v001 import create_h2o_urd_model
from DNN_h2o_functions_v001 import get_predictions


if __name__ == "__main__":

    # Define home directory
    home_path = None
    if platform.system() == 'Linux':
        home_path = os.path.expanduser("~")
    elif platform.system() == 'Windows':
        home_path = 'C:\\'

    # Import URD data
    urd_path = os.path.join(home_path, '0MyDataBases/40Python/URD_DNN/data/ExportFileR.csv')
    data_full = pd.read_csv(urd_path)

    # Create H2O model
    model = create_h2o_urd_model(data_full)

    # Define list of pandas DataFrames for model to predict on
    base_data_path = os.path.join(home_path, '0MyDataBases/40Python/URD_DNN/data')
    l_csv_test_data = ['ExportFileWeather_2015.csv', 'ExportFileWeather_2014.csv', 'ExportFileWeather_2013.csv',
                       'ExportFileWeather_2012.csv', 'ExportFileWeather_2011.csv', 'ExportFileWeather_2010.csv']
    l_pd_test_data = [data_full]
    for csv_test_data in l_csv_test_data:
        l_pd_test_data.append(pd.read_csv(os.path.join(base_data_path, csv_test_data)))

    # Get model predictions on test data
    l_predictions_raw = get_predictions(model, l_pd_test_data, ['ExportFileR.csv'] + l_csv_test_data)

    # Add prediction column to existing pandas DataFrames
    for i in range(len(l_predictions_raw)):
        l_pd_test_data[i]['Prediction'] = l_predictions_raw[i]['predict']

    # Create list of strings indicating year of test data being used
    l_test_year = ['.Actual']
    for filename in sorted(l_csv_test_data):
        l_test_year.append(filename[-8:-4])

    # Zip weather data year with predictions and errors
    d_years_predictions = {}
    for i in range(len(l_test_year)):
        d_years_predictions[l_test_year[i]] = l_pd_test_data[i]

    # Plot using external .py file function
    visualize_urd(data_full, d_years_predictions, aggregations=set(['Daily', 'Yearly', 'Monthly']))

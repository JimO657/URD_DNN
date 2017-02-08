import pandas as pd
import os
import platform

from DNN_plotly_functions_v001 import visualize_urd
from DNN_h2o_functions_v001 import create_h2o_urd_model, get_predictions
from DNN_highcharts_functions_v001 import make_highcharts, visualize_urd_highcharts


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

    # Create H2O model using external .py file function
    model = create_h2o_urd_model(data_full)

    # Define list of pandas DataFrames for model to predict on
    base_data_path = os.path.join(home_path, '0MyDataBases/40Python/URD_DNN/data')
    l_csv_test_data = []  # 'ExportFileWeather_2015.csv', 'ExportFileWeather_2014.csv', 'ExportFileWeather_2013.csv',
                          # 'ExportFileWeather_2012.csv', 'ExportFileWeather_2011.csv', 'ExportFileWeather_2010.csv']
    for i in range(2000, 2016):
        l_csv_test_data.append('ExportFileWeather_{}.csv'.format(str(i)))

    # Create dictionary mapping labels for weather years to corresponding pandas DataFrames
    d_pd_test_data = {'.Actual': data_full}
    for csv_test_data in l_csv_test_data:
        d_pd_test_data[csv_test_data[-8:-4]] = pd.read_csv(os.path.join(base_data_path, csv_test_data))

    # Get model predictions on test data, put directly in pandas DataFrames inside dictionary
    d_pd_test_data = get_predictions(model, d_pd_test_data)

    # Plot with plotly using external .py file function
    # visualize_urd(data_full, d_pd_test_data, aggregations={'Yearly'})

    # Plot with highcharts using external .py file function
    visualize_urd_highcharts(data_full, d_pd_test_data)

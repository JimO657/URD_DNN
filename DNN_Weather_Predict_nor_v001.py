import pandas as pd
import os
import platform

from DNN_plotly_functions_v001 import visualize_urd
from DNN_h2o_functions_v001 import create_h2o_urd_model, get_predictions
from DNN_highcharts_functions_v001 import make_highcharts, visualize_urd_highcharts


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
model = create_h2o_urd_model(data_full,ip="192.168.0.21")

# Define list of pandas DataFrames for model to predict on
base_data_path = os.path.join(home_path, '0MyDataBases/40Python/URD_DNN/data')
l_csv_test_data = []  # 'ExportFileWeather_2015.csv', 'ExportFileWeather_2014.csv', 'ExportFileWeather_2013.csv',
                      # 'ExportFileWeather_2012.csv', 'ExportFileWeather_2011.csv', 'ExportFileWeather_2010.csv']
for i in range(2000, 2016):
    l_csv_test_data.append('ExportFileWeather_{}.csv'.format(str(i)))

# Create dictionary mapping labels for weather years to corresponding pandas DataFrames
d_pd_test_data = {'2016': data_full}
for csv_test_data in l_csv_test_data:
    d_pd_test_data[csv_test_data[-8:-4]] = pd.read_csv(os.path.join(base_data_path, csv_test_data))

# Get model predictions on test data, put directly in pandas DataFrames inside dictionary
d_pd_test_data = get_predictions(model, d_pd_test_data)

########################################################
# Creating and exporting the full prediction csv
del edf
edf=d_pd_test_data[str(2000)].iloc [:,[0,1]]

for key in d_pd_test_data.keys():
  # print "the key name is " + key
  edf.loc[:,key]=d_pd_test_data[key].loc [:,"Prediction"]
edf.to_csv("data/predictions.csv",index=False)


######################################################
# Plot with plotly using external .py file function
visualize_urd(data_full, d_pd_test_data, aggregations={'Yearly','Monthly','Daily'})

# Plot with highcharts using external .py file function
# highcharts_path = os.path.join(home_path, '0MyDataBases/40Python/URD_DNN/Highcharts')
# visualize_urd_highcharts(data_full, d_pd_test_data, highcharts_path)
#
#
# """
# For documentation of the webbrowser module,
# see http://docs.python.org/library/webbrowser.html
# """
# import webbrowser
# new = 2 # open in a new tab, if possible
#
# # open a public URL, in this case, the webbrowser docs
# url = "file:///C:/0MyDataBases/40Python/URD_DNN/Highcharts/URD_Prediction_2000_Weather.html"
# webbrowser.open(url,new=new)
#

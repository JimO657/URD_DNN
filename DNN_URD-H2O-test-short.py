import pandas as pd
import os
import platform

from DNN_plotly_functions_v001 import visualize_urd
from DNN_h2o_functions_v001 import create_h2o_urd_model, get_predictions
from DNN_highcharts_functions_v001 import make_highcharts, visualize_urd_highcharts
import pandas as pd
import h2o
from h2o import exceptions
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators import H2ODeepWaterEstimator
import platform
import sys
from select import select
from getpass import getuser
# import time


##### Start h2o#######
#h2o.init(nthreads=71, max_mem_size='30G')
h2o.init(ip="192.168.0.11",strict_version_check=False)

# Remove all objects from h2o
#h2o.remove_all()

###################################

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
model = create_h2o_urd_model(data_full,epochs=50, hidden=[80, 80], stopping_rounds=5)



# Define list of pandas DataFrames for model to predict on
base_data_path = os.path.join(home_path, '0MyDataBases/40Python/URD_DNN/data')
l_csv_test_data = ['ExportFileWeather_2015.csv', 'ExportFileWeather_2014.csv', 'ExportFileWeather_2013.csv',
                   'ExportFileWeather_2012.csv', 'ExportFileWeather_2011.csv', 'ExportFileWeather_2010.csv']

# Create dictionary mapping labels for weather years to corresponding pandas DataFrames
d_pd_test_data = {'.Actual': data_full}
for csv_test_data in l_csv_test_data:
    d_pd_test_data[csv_test_data[-8:-4]] = pd.read_csv(os.path.join(base_data_path, csv_test_data))

# Get model predictions on test data, put directly in pandas DataFrames inside dictionary
d_pd_test_data = get_predictions(model, d_pd_test_data)


###############################################
# Plot with plotly using external .py file function
visualize_urd(data_full, d_pd_test_data, aggregations={'Yearly'})

# Plot with highcharts using external .py file function
#visualize_urd_highcharts(data_full, d_pd_test_data)

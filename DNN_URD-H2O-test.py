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
import time


def write_log(str):
    f_path = 'C:/0MyDataBases/40Python/URD_DNN/logs/P_log.log'
    current_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time()))
    with open(f_path, 'w') as f:
        f.write(str+ " " +current_time)
    return

write_log("start script")
##### Start h2o#######
#h2o.init(nthreads=71, max_mem_size='30G')
#h2o.init(ip="192.168.0.11",strict_version_check=False)

# Remove all objects from h2o
#h2o.remove_all()

def create_h2o_urd_model_l(urd_data, urd_model_id='Python_URD_DNN_2006-2014' + getuser() + platform.system(), ip="localhost", epochs=5000, hidden=[800, 800], stopping_rounds=5):
    """Creates an H2O model from URD data

    Args:
        urd_data (pandas DataFrame): URD data as DataFrame.
        urd_model_id (string): Name of model. Defaults to 'Python_URD_DNN_2006-2014[USER][OS]'
        ip (string, optional): IP of H2O cluster to connect to
        epochs (float, optional): Number of epochs to pass to H2O estimator. Defaults to 5000.
        hidden (list, optional): Layers to pass to H2O estimator. Defaults to [800, 800]
        stopping_rounds (int, optional): Number of stopping rounds to pass to H2O estimator. Defaults to 5.

    Returns:
        H2ODeepLearningEstimator: URD model.

    """

    # Start H2O
    h2o.init(ip=ip, strict_version_check=False)

    # Get user
    user = getuser()

    # Get current time for timestamp (currently not used)
    # current_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time()))

    # Define path to model
    # urd_model_id = 'Python_URD_DNN_2006-2014' + user + platform.system()
    save_path = 'H2O_Models/'

    # Try to load existing model
    try:
        existing_model = h2o.load_model(save_path + urd_model_id)
    except exceptions.H2OResponseError:
        existing_model = False

    # If model exists, prompt user
    if existing_model:

        # Create skip prompt and variable
        prompt_for_skip = """Found H2O model {}. Use existing model? ('n' to build new model, anything
                             else or wait 5 seconds to continue): """.format(urd_model_id)

        # Check if model exists and prompt for overwrite if exists with timeout - LINUX ONLY
        if platform.system() == 'Linux':
            timeout = 5
            print(prompt_for_skip),
            rlist, _, _ = select([sys.stdin], [], [], timeout)
            if rlist:
                skip_h2o = sys.stdin.readline()[0:-1]  # Remove newline at end
            else:
                skip_h2o = ""
                print("\n")

        # Skip prompt and force create model
        elif platform.system() == 'Windows':
            # skip_h2o = raw_input(prompt_for_skip)
            skip_h2o = 'n'

    else:
        skip_h2o = 'n'

    if skip_h2o != 'n':  # Return existing model

        urd_model = existing_model

    else:  # Create new model

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
                             column_types=['int', 'enum', 'real', 'real', 'int', 'int', 'int', 'int'],
                             destination_frame='Training_Validation_Frame')
        training, validation = train.split_frame(ratios=[0.8])

        # Define predictors and response
        predictors = list(train.columns)[1:]
        response = list(train.columns)[0]

        # Run DNN
        urd_model = H2ODeepLearningEstimator(model_id=urd_model_id, epochs=epochs, hidden=hidden, activation="Tanh",
                                             l1=0, l2=0, stopping_rounds=stopping_rounds, stopping_metric='MSE',
                                             stopping_tolerance=1e-6)

        urd_model.train(x=predictors, y=response, training_frame=training, validation_frame=validation)

        # Save DNN model to /tmp folder
        h2o.save_model(urd_model, path=save_path, force=True)

    return urd_model





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

write_log("about to start training")
# Create H2O model using external .py file function
model = create_h2o_urd_model_l(data_full, urd_model_id="Test-URD-MODEL" , epochs=500, hidden=[800, 800, 800], stopping_rounds=0)
write_log("finished training")


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
visualize_urd(data_full, d_pd_test_data, aggregations={'Yearly','Monthly','Daily'})

# Plot with highcharts using external .py file function
#visualize_urd_highcharts(data_full, d_pd_test_data)

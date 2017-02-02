import pandas as pd
import h2o
import os
from h2o import exceptions
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators import H2ODeepWaterEstimator
import platform
import sys
from select import select


def create_h2o_urd_model(urd_data, epochs=5000, hidden=[800, 800], stopping_rounds=5):
    """Creates an H2O model from URD data

    Args:
        urd_data (pandas DataFrame): URD data as DataFrame.
        epochs (float, optional): Number of epochs to pass to H2O estimator. Defaults to 5000.
        hidden (list, optional): Layers to pass to H2O estimator. Defaults to [800, 800]
        stopping_rounds (int, optional): Number of stopping rounds to pass to H2O estimator. Defaults to 5.

    Returns:
        H2ODeepLearningEstimator: URD model.

    """

    # Start H2O and remove all objects
    h2o.init(strict_version_check=False)
    # h2o.remove_all()

    # Define path to model
    urd_model_id = 'Python_URD_DNN_2006-2014'
    save_path = None
    if platform.system() == 'Linux':
        save_path = os.path.join(os.environ.get('HOME'), '0MyDataBases/7R/ADHOC_Qlikview-linux/H2O_Models_2015/')
    elif platform.system() == 'Windows':
        save_path = 'C:\\from-linux\\0MyDataBases\\7R\ADHOC_Qlikview-linux\H2O_Models\\'

    # Create skip prompt and variable
    prompt_for_skip = """H2O model {} already exists. Use existing model? ('n' to build new model, anything
                         else or wait 5 seconds to continue): """.format(urd_model_id)
    skip_h2o = None

    # Check if model exists
    if os.path.exists(os.path.join(save_path, urd_model_id)):

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

        # Check if model exists and prompt for overwrite WITHOUT timeout
        elif platform.system() == 'Windows':
            skip_h2o = raw_input(prompt_for_skip)
    else:
        skip_h2o = 'n'

    # Load or create new model
    if skip_h2o != 'n':  # Load model
        urd_model = h2o.load_model(os.path.join(save_path, urd_model_id))

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
        predictors = list(train.columns)[2:]
        response = list(train.columns)[0]

        # Run DNN
        urd_model = H2ODeepLearningEstimator(model_id=urd_model_id, epochs=epochs, hidden=hidden, activation="Tanh",
                                         l1=0, l2=0, stopping_rounds=stopping_rounds, stopping_metric='MSE', stopping_tolerance=1e-6)
        # urd_model = H2ODeepWaterEstimator(model_id=urd_model_id, epochs=5000, hidden=[800, 800], activation="Tanh",
        #                                      stopping_rounds=5, stopping_metric='MSE', stopping_tolerance=1e-6)
        urd_model.train(x=predictors, y=response, training_frame=training, validation_frame=validation)

        # Save DNN model
        try:
            h2o.save_model(urd_model, path=save_path)
        except exceptions.H2OServerError:  # Raised if file already exists
            os.remove(save_path + urd_model_id)
            h2o.save_model(urd_model, path=save_path)

    return urd_model


def get_predictions(urd_model, l_test_data, l_test_names):
    """Creates predictions on test data using H2O model

    Args:
        urd_model (H2ODeepLearningEstimator): H2O model for prediction.
        l_test_data (list): list of pandas DataFrames to predict on.
        l_test_names (list): list of csv files corresponding to the DataFrames

    Returns:
        list: List of pandas DataFrames containing predictions.

    """

    # Initialize list of predictions
    predictions = []

    # Convert test data to H2OFrames and run predictions
    for i in range(len(l_test_data)):
        test_data = l_test_data[i]
        test_data_no_date = test_data.drop('Date1', axis=1, inplace=False)
        h2o_test = h2o.H2OFrame(test_data_no_date,
                                column_types=['int', 'enum', 'real', 'real', 'int', 'int', 'int', 'int'],
                                destination_frame=l_test_names[i] + "_Prediction")
        predictions.append(urd_model.predict(h2o_test).as_data_frame())

    return predictions

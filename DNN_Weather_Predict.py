import pandas as pd
import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator


if __name__=='__main__':

    # Start h2o
    h2o.init(nthreads=71, max_mem_size='30G')

    # Remove all objects from h2o
    h2o.remove_all()

    # Import data to pandas dataframe
    data_full = pd.read_csv('data_2015/ExportFileR.csv')

    # Set start and end dates for training
    date_start = '2006-Jan-01'
    date_end = '2014-Dec-31'

    # Find row indices of training data
    start_row = data_full[data_full['Date1'] == date_start].index.tolist()[0]
    end_row = data_full[data_full['Date1'] == date_end].index.tolist()[0]

    # Create training data slice and convert to H2OFrame
    train_pd = data_full[start_row:end_row+1]
    train = h2o.H2OFrame(train_pd)
    training, validation = train.split_frame(ratios=[0.8])

    # Create test data slice and convert to H2OFrame
    test_pd = data_full[end_row + 1:]
    test = h2o.H2OFrame(test_pd)

    # Define predictors and output
    predictors = list(train.columns)[2:]
    output = list(train.columns)[0]

    # Run DNN
    model_id = 'Python_URD_DNN_2006-2014'
    model = H2ODeepLearningEstimator(model_id=model_id, epochs=50, hidden=[800,800], activation="Tanh",
                                     l1=0, l2=0, score_training_samples=5, score_validation_samples=5)
    model.train(x=predictors, y=output, training_frame=training, validation_frame=validation)


import pandas as pd
import h2o
import os
import plotly
import plotly.graph_objs as go
from datetime import datetime
from tqdm import tqdm


def aggregate_by_day_month_year(dataframe):
    """Aggregates pandas DataFrames by day, month, and year using indices.

    Args:
        dataframe (pandas DataFrame): DataFrame with 'Date1' column.

    Returns:
        dictionary: Maps words 'Daily', 'Monthly', and 'Yearly' to aggregated pandas DataFrame.

    """

    import pandas as pd

    # Create time index
    times = pd.DatetimeIndex(dataframe.Date1)

    # Create daily aggregate
    pd_daily = dataframe.groupby([times.year, times.month, times.day]).sum() # Aggregate by day
    pd_daily.reset_index(inplace=True)  # Turns multi index into columns
    pd_daily = pd_daily.rename(columns={'level_0': 'Year', 'level_1': 'Month', 'level_2': 'Day'})
    pd_daily['Date'] = pd_daily.apply(lambda row: datetime(int(row['Year']), int(row['Month']), int(row['Day']), 1), axis=1)

    # Create monthly aggregate
    pd_monthly = dataframe.groupby([times.year, times.month]).sum() # Aggregate by month
    pd_monthly.reset_index(inplace=True)  # Turns multi index into columns
    pd_monthly = pd_monthly.rename(columns={'level_0': 'Year', 'level_1': 'Month'})  # Rename index columns
    pd_monthly['Date'] = pd_monthly.apply(lambda row: datetime(int(row['Year']), int(row['Month']), 1), axis=1)

    # Create yearly aggregate
    pd_yearly = dataframe.groupby([times.year]).sum()
    pd_yearly.reset_index(inplace=True)
    pd_yearly = pd_yearly.rename(columns={'index': 'Date'})

    # Create error columns
    pd_daily['Error'] = pd_daily['Prediction'] - pd_daily['ACT']
    pd_monthly['Error'] = pd_monthly['Prediction'] - pd_monthly['ACT']
    pd_yearly['Error'] = pd_yearly['Prediction'] - pd_yearly['ACT']

    return {'Daily': pd_daily, 'Monthly': pd_monthly, 'Yearly': pd_yearly}


def make_scatter_trace(column1, column2, name='', legendgroup=''):
    """Creates plotly trace from pandas dataframe columns.

    Args:
        column1 (pandas DataFrame): Single-column DataFrame to use as x-axis.
        column2 (pandas DataFrame): Single-column DataFrame to use as y-axis.
        name (str, opt.): Name of scatter trace.
        legendgroup (str, opt.): Name of legend group to put trace in.

    Returns:
        Scatter plot

    """

    # Import libraries
    import pandas as pd
    import plotly
    import plotly.graph_objs as go

    # Create scatter plot
    scatter_trace = go.Scatter(
        x=column1,
        y=column2,
        name=name,
        legendgroup=legendgroup,
    )

    return scatter_trace


if __name__ == "__main__":

    # Define home directory
    home_path = os.path.expanduser("~")

    # Import URD data
    urd_path = os.path.join(home_path, '0MyDataBases/7R/ADHOC_Qlikview-linux/data_2015/ExportFileR.csv')
    data_full = pd.read_csv(urd_path)

    # Create H2O model
    model = create_h2o_urd_model(data_full)

    # Define list of pandas DataFrames for model to predict on
    base_data_path = os.path.join(home_path, '0MyDataBases/7R/ADHOC_Qlikview-linux/data_2015')
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

    # Aggregate full data into yearly, monthly, and daily results with errors
    l_predictions = []
    for pd_test_data in tqdm(l_pd_test_data):
        l_predictions.append(aggregate_by_day_month_year(pd_test_data))

    # Create list of strings indicating year of test data being used
    l_test_year = ['Actual']
    for filename in l_csv_test_data:
        l_test_year.append(filename[-8:-4])

    # Zip weather data year with predictions and errors
    d_predictions = {}
    for i in range(len(l_test_year)):
        d_predictions[l_test_year[i]] = l_predictions[i]

    # Create traces for predictions
    d_traces = {}
    for weather_year in d_predictions:
        for time_frame in d_predictions[weather_year]:
            try:
                d_traces[weather_year][time_frame] = {}
                d_traces[weather_year][time_frame]['Prediction'] = go.Scatter(
                    x=d_predictions[weather_year][time_frame]['Date'],
                    y=d_predictions[weather_year][time_frame]['Prediction'],
                    name="{0} Prediction ({1} Weather)".format(time_frame, weather_year),
                    line=dict(dash='dash'),
                    legendgroup=time_frame
                )
                d_traces[weather_year][time_frame]['Error'] = go.Bar(
                    x=d_predictions[weather_year][time_frame]['Date'],
                    y=d_predictions[weather_year][time_frame]['Error'],
                    name="{0} Error ({1} Weather)".format(time_frame, weather_year),
                    legendgroup=time_frame,
                    yaxis='y2'
                )

            except KeyError:
                d_traces[weather_year] = {}
                d_traces[weather_year][time_frame] = {}
                d_traces[weather_year][time_frame]['Prediction'] = go.Scatter(
                    x=d_predictions[weather_year][time_frame]['Date'],
                    y=d_predictions[weather_year][time_frame]['Prediction'],
                    name="{0} Prediction ({1} Weather)".format(time_frame, weather_year),
                    line=dict(dash='dash'),
                    legendgroup=time_frame
                )
                d_traces[weather_year][time_frame]['Error'] = go.Bar(
                    x=d_predictions[weather_year][time_frame]['Date'],
                    y=d_predictions[weather_year][time_frame]['Error'],
                    name="{0} Error ({1} Weather)".format(time_frame, weather_year),
                    legendgroup=time_frame,
                    yaxis='y2'
                )

    # Create traces for actual data
    trace_yearly = go.Scatter(
        x=d_predictions['Actual']['Yearly']['Date'],
        y=d_predictions['Actual']['Yearly']['ACT'],
        name='Yearly Actual',
        legendgroup='Yearly',
    )
    trace_monthly = go.Scatter(
        x=d_predictions['Actual']['Monthly']['Date'],
        y=d_predictions['Actual']['Monthly']['ACT'],
        name='Monthly Actual',
        legendgroup='Monthly',
    )
    trace_daily = go.Scatter(
        x=d_predictions['Actual']['Daily']['Date'],
        y=d_predictions['Actual']['Daily']['ACT'],
        name='Daily Actual',
        legendgroup='Daily',
    )

    # Create data and visibility dictionaries
    l_vis_dicts = []
    for test_year in l_test_year:
        visibility = [True, True, True]
        l_data = [trace_yearly, trace_monthly, trace_daily]
        for weather_year in d_traces:
            for time_frame in d_traces[weather_year]:
                for data_type in d_traces[weather_year][time_frame]:
                    l_data.append(d_traces[weather_year][time_frame][data_type])
                    if weather_year == test_year:
                        visibility.append(True)
                    else:
                        visibility.append(False)
        l_vis_dicts.append({'args': ['visible', visibility],
                            'label': test_year,
                            'method': 'restyle'
                            })

        data = go.Data(l_data)

    # Create layout for plotly
    layout = go.Layout(
        title='URD Prediction vs. Actual',
        xaxis=dict(title='', rangeslider=dict(thickness=0.015, borderwidth=1), type='date', showgrid=True),
        yaxis=dict(title='', showgrid=True, domain=[0.35, 1]),
        yaxis2=dict(domain=[0, 0.25]),
        updatemenus=list([
            dict(
                buttons=[vis_dict for vis_dict in l_vis_dicts],
                type='buttons'
            )]))

    # Plot with plotly
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig)

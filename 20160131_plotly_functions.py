import pandas as pd
import plotly
import plotly.graph_objs as go
from datetime import datetime
from tqdm import tqdm
import sys


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


def visualize_urd(real_data, predictions, filename='temp_plot.html'):
    """Creates html file to visualize real data and predictions for URD data using plotly.

    Args:
        real_data (pandas DataFrame): Pandas dataframe containing real data with 'Date1' and 'ACT' columns.
        predictions (dictionary): Dictionary mapping years as strings to pandas dataframes containing predictions with
            'Date1' and 'Prediction' columns.
        filename (str, opt.): Path to html file to be written.

    Returns:
        html file

    """

    # Create dictionary of aggregation type to actual dataframe
    d_actual = aggregate_by_day_month_year(real_data)

    # Create nested dictionary of applied weather year to aggregation type to prediction dataframe
    d_predictions = {}
    print("Aggregating data...")
    sys.stdout.write('.')
    sys.stdout.flush()
    for prediction_year in tqdm(predictions):
        d_predictions[prediction_year] = aggregate_by_day_month_year(predictions[prediction_year])

    # Create traces for predictions
    d_traces = {}
    for weather_year in sorted(d_predictions):
        for time_frame in d_predictions[weather_year]:
            try:
                d_traces[weather_year][time_frame] = {}
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
        x=d_actual['Yearly']['Date'],
        y=d_actual['Yearly']['ACT'],
        name='Yearly Actual',
        legendgroup='Yearly',
    )
    trace_monthly = go.Scatter(
        x=d_actual['Monthly']['Date'],
        y=d_actual['Monthly']['ACT'],
        name='Monthly Actual',
        legendgroup='Monthly',
    )
    trace_daily = go.Scatter(
        x=d_actual['Daily']['Date'],
        y=d_actual['Daily']['ACT'],
        name='Daily Actual',
        legendgroup='Daily',
    )

    # Create data and visibility dictionaries
    l_vis_dicts = []
    for test_year in sorted(d_predictions):
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
    plotly.offline.plot(fig, filename=filename)

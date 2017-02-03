import pandas as pd
import plotly
import plotly.graph_objs as go
from datetime import datetime
from tqdm import tqdm


def aggregate_by_day_month_year(dataframe, aggregations):
    """Aggregates pandas DataFrames by day, month, and year using indices.

    Args:
        dataframe (pandas DataFrame): DataFrame with 'Date1' column.
        aggregations (set): set of strings defining which aggregations (Yearly, Monthly, Daily) to use

    Returns:
        dictionary: Maps words 'Daily', 'Monthly', and 'Yearly' to aggregated pandas DataFrame.

    """

    # Initialize dictionary to be returned
    return_dict = {}

    # Create time index
    times = pd.DatetimeIndex(dataframe.Date1)

    # Create daily aggregate with error column
    if 'Daily' in aggregations:
        pd_daily = dataframe.groupby([times.year, times.month, times.day]).sum() # Aggregate by day
        pd_daily.reset_index(inplace=True)  # Turns multi index into columns
        pd_daily = pd_daily.rename(columns={'level_0': 'Year', 'level_1': 'Month', 'level_2': 'Day'})
        pd_daily['Date'] = pd_daily.apply(lambda row: datetime(int(row['Year']), int(row['Month']), int(row['Day']), 1), axis=1)
        pd_daily['Error'] = pd_daily['Prediction'] - pd_daily['ACT']

        return_dict['Daily'] = pd_daily

    # Create monthly aggregate with error column
    if 'Monthly' in aggregations:
        pd_monthly = dataframe.groupby([times.year, times.month]).sum() # Aggregate by month
        pd_monthly.reset_index(inplace=True)  # Turns multi index into columns
        pd_monthly = pd_monthly.rename(columns={'level_0': 'Year', 'level_1': 'Month'})  # Rename index columns
        pd_monthly['Date'] = pd_monthly.apply(lambda row: datetime(int(row['Year']), int(row['Month']), 1), axis=1)
        pd_monthly['Error'] = pd_monthly['Prediction'] - pd_monthly['ACT']

        return_dict['Monthly'] = pd_monthly

    # Create yearly aggregate with error column
    if 'Yearly' in aggregations:
        pd_yearly = dataframe.groupby([times.year]).sum()
        pd_yearly.reset_index(inplace=True)
        pd_yearly = pd_yearly.rename(columns={'index': 'Date'})
        pd_yearly['Error'] = pd_yearly['Prediction'] - pd_yearly['ACT']

        return_dict['Yearly'] = pd_yearly

    return return_dict


def visualize_urd(real_data, predictions, filename='temp_plot.html', aggregations=set(['Yearly'])):
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
    d_actual = aggregate_by_day_month_year(real_data, aggregations)

    # Create nested dictionary of applied weather year to aggregation type to prediction dataframe
    d_predictions = {}
    print("Aggregating data...")
    for prediction_year in tqdm(predictions):
        d_predictions[prediction_year] = aggregate_by_day_month_year(predictions[prediction_year], aggregations)

    # Create traces for predictions
    d_traces = {}
    for weather_year in sorted(d_predictions):
        for time_frame in d_predictions[weather_year]:
            try:
                d_traces[weather_year][time_frame] = {}
            except KeyError:
                d_traces[weather_year] = {}
                d_traces[weather_year][time_frame] = {}

            # Define y-axis variables, dependent on time frame and error
            if time_frame == 'Yearly':
                yaxis_prediction = 'y1'
                yaxis_error = 'y2'
            elif time_frame == 'Monthly':
                yaxis_prediction = 'y3'
                yaxis_error = 'y4'
            elif time_frame == 'Daily':
                yaxis_prediction = 'y5'
                yaxis_error = 'y6'

            d_traces[weather_year][time_frame]['Prediction'] = go.Scatter(
                x=d_predictions[weather_year][time_frame]['Date'],
                y=d_predictions[weather_year][time_frame]['Prediction'],
                name="{0} Prediction ({1} Weather)".format(time_frame, weather_year),
                line=dict(dash='dash'),
                legendgroup=time_frame,
                yaxis=yaxis_prediction,
                visible=False
            )
            d_traces[weather_year][time_frame]['Error'] = go.Bar(
                x=d_predictions[weather_year][time_frame]['Date'],
                y=d_predictions[weather_year][time_frame]['Error'],
                name="{0} Error ({1} Weather)".format(time_frame, weather_year),
                legendgroup=time_frame,
                yaxis=yaxis_error,
                visible=False
            )

    # Create traces for actual data
    if 'Yearly' in aggregations:
        trace_yearly = go.Scatter(
            x=d_actual['Yearly']['Date'],
            y=d_actual['Yearly']['ACT'],
            name='Yearly Actual',
            legendgroup='Yearly',
        )
    if 'Monthly' in aggregations:
        trace_monthly = go.Scatter(
            x=d_actual['Monthly']['Date'],
            y=d_actual['Monthly']['ACT'],
            name='Monthly Actual',
            legendgroup='Monthly',
        )
    if 'Daily' in aggregations:
        trace_daily = go.Scatter(
            x=d_actual['Daily']['Date'],
            y=d_actual['Daily']['ACT'],
            name='Daily Actual',
            legendgroup='Daily',
        )

    # Create data and visibility dictionaries
    l_vis_dicts = []
    for test_year in sorted(d_predictions):
        visibility = []
        l_data = []
        for aggregation in sorted(aggregations):
            visibility.append(True)
            if aggregation == 'Yearly':
                l_data.append(trace_yearly)
            elif aggregation == 'Monthly':
                l_data.append(trace_monthly)
            elif aggregation == 'Daily':
                l_data.append(trace_daily)
        for weather_year in sorted(d_traces):
            for time_frame in sorted(d_traces[weather_year]):
                for data_type in sorted(d_traces[weather_year][time_frame]):
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
        yaxis=dict(title='', showgrid=True, domain=[0.85, 1]),
        yaxis2=dict(domain=[0.7, 0.8]),
        yaxis3=dict(domain=[0.5, 0.65]),
        yaxis4=dict(domain=[0.35, 0.45]),
        yaxis5=dict(domain=[0.15, 0.3]),
        yaxis6=dict(domain=[0, 0.1]),
        updatemenus=list([
            dict(
                buttons=[vis_dict for vis_dict in l_vis_dicts],
                type='buttons'
            )]))

    # Plot with plotly
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=filename)

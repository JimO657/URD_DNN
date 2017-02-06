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


def assign_domains(aggregations, padding=0.05, scatter_ratio = 0.75):
    """Defines y-axis and domains for plotly layout

    Args:
        aggregations (set): Aggregations (Daily, Monthly, Yearly) to use in calculations.
        padding (float, opt.): Space between plots as fraction of full plot's height. Default 0.05.
        scatter_ratio (float, opt.): Fraction of each aggregate set of subplots dedicated to line subplot. Default 0.75.

    Returns:
        nested dictionary of structure: {Aggregation: {'Prediction' or 'Error': {'Axis':'#', 'Bounds': [#, #]}}}

    """

    length = 2 * len(aggregations)
    bar_ratio = 1 - scatter_ratio
    scatter_thickness = (1 - padding * (length - 1)) / length * 2 * scatter_ratio
    bar_thickness = (1 - padding * (length - 1)) / length * 2 * bar_ratio

    axis_bounds = {}
    counter = 0  # Counter variable for assigning axes
    bound = 0  # Bound variable

    for agg in sorted(aggregations):

        axis_bounds[agg] = {'Prediction': {}, 'Error': {}}

        lower_bound = bound
        bound += bar_thickness
        upper_bound = bound

        axis_bounds[agg]['Error']['Axis'] = str(counter + 1)
        axis_bounds[agg]['Error']['Bounds'] = [lower_bound, upper_bound]

        bound += padding
        lower_bound = bound
        bound += + scatter_thickness
        upper_bound = bound

        axis_bounds[agg]['Prediction']['Axis'] = str(counter + 2)
        axis_bounds[agg]['Prediction']['Bounds'] = [lower_bound, upper_bound]

        bound += padding
        counter += 2

    return axis_bounds


def visualize_urd(real_data, predictions, filename='temp_plot.html', aggregations=set(['Yearly'])):
    """Creates html file to visualize real data and predictions for URD data using plotly.

    Args:
        real_data (pandas DataFrame): Pandas dataframe containing real data with 'Date1' and 'ACT' columns.
        predictions (dictionary): Dictionary mapping years as strings to pandas dataframes containing predictions with
            'Date1' and 'Prediction' columns.
        filename (str, opt.): Path to html file to be written. Default 'temp.plot.html'.
        aggregations (set, opt.): Aggregations (Daily, Monthly, Yearly) to calculate and plot. Default {'Yearly'}.

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

    # Define axes and bounds to be used in plotly based on aggregations
    d_domains = assign_domains(aggregations)

    # Create traces for predictions
    d_traces = {}
    for weather_year in sorted(d_predictions):

        # Make initial visibility variable true if prediction is for actual weather
        initial_visibility = False
        if weather_year == '.Actual':
            initial_visibility = True

        for time_frame in d_predictions[weather_year]:
            try:
                d_traces[weather_year][time_frame] = {}
            except KeyError:
                d_traces[weather_year] = {}
                d_traces[weather_year][time_frame] = {}

            # Define y-axis variables
            yaxis_error = 'y' + d_domains[time_frame]['Error']['Axis']
            yaxis_prediction = 'y' + d_domains[time_frame]['Prediction']['Axis']

            # Create scatter trace for prediction
            d_traces[weather_year][time_frame]['Prediction'] = go.Scatter(
                x=d_predictions[weather_year][time_frame]['Date'],
                y=d_predictions[weather_year][time_frame]['Prediction'],
                name="{0} Prediction ({1} Weather)".format(time_frame, weather_year),
                line=dict(dash='dash', color='#A00'),
                legendgroup=time_frame,
                yaxis=yaxis_prediction,
                visible=initial_visibility,
            )

            # Create bar trace for error
            d_traces[weather_year][time_frame]['Error'] = go.Bar(
                x=d_predictions[weather_year][time_frame]['Date'],
                y=d_predictions[weather_year][time_frame]['Error'],
                name="{0} Error ({1} Weather)".format(time_frame, weather_year),
                marker=dict(color='#777'),
                legendgroup=time_frame,
                yaxis=yaxis_error,
                visible=initial_visibility,
            )

    # Create traces for actual data
    if 'Yearly' in aggregations:
        trace_yearly = go.Scatter(
            x=d_actual['Yearly']['Date'],
            y=d_actual['Yearly']['ACT'],
            name='Yearly Actual',
            line=dict(color='#00A'),
            legendgroup='Yearly',
            yaxis='y' + d_domains['Yearly']['Prediction']['Axis'],
        )
    if 'Monthly' in aggregations:
        trace_monthly = go.Scatter(
            x=d_actual['Monthly']['Date'],
            y=d_actual['Monthly']['ACT'],
            name='Monthly Actual',
            line=dict(color='#00A'),
            legendgroup='Monthly',
            yaxis='y' + d_domains['Monthly']['Prediction']['Axis'],
        )
    if 'Daily' in aggregations:
        trace_daily = go.Scatter(
            x=d_actual['Daily']['Date'],
            y=d_actual['Daily']['ACT'],
            name='Daily Actual',
            line=dict(color='#00A'),
            legendgroup='Daily',
            yaxis='y' + d_domains['Daily']['Prediction']['Axis'],
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

    # Create axes based on aggregations passed
    d_axis_domains = {}

    for agg in d_domains:
        for data_type in d_domains[agg]:
            d_axis_domains['yaxis' + d_domains[agg][data_type]['Axis']] = dict(
                                                                            domain=d_domains[agg][data_type]['Bounds'])

    # Create layout for plotly
    layout = dict(
        title='URD Prediction vs. Actual',
        xaxis1=dict(title='', rangeslider=dict(thickness=0.015, borderwidth=1), type='date', showgrid=True),
        updatemenus=list([
            dict(
                buttons=[vis_dict for vis_dict in l_vis_dicts],
                type='buttons',
                active=0,
            )]))

    layout.update(d_axis_domains)  # Update layout with yaxis# keys

    # Plot with plotly
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=filename)

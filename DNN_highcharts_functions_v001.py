import pandas as pd
from datetime import datetime
from tqdm import tqdm
from highcharts import Highchart


def aggregate_by_day_month_year(dataframe, aggregations, date_column_name='Date'):
    """Aggregates pandas DataFrames by day, month, and year using indices.

    Args:
        dataframe (pandas DataFrame): DataFrame with column that can be converted to pandas DatetimeIndex.
        aggregations (set): set of strings defining which aggregations (Yearly, Monthly, Daily) to use.
        date_column_name (string): Name of dataframe column to be converted to DatetimeIndex.

    Returns:
        dictionary: Maps words 'Daily', 'Monthly', and 'Yearly' to aggregated pandas DataFrame.

    """

    # Initialize dictionary to be returned
    return_dict = {}

    # Create time index
    times = pd.DatetimeIndex(dataframe[date_column_name])

    # Create daily aggregate with error column
    if 'Daily' in aggregations:
        pd_daily = dataframe.groupby([times.year, times.month, times.day]).sum() # Aggregate by day
        pd_daily.reset_index(inplace=True)  # Turns multi index into columns
        pd_daily = pd_daily.rename(columns={'level_0': 'Year', 'level_1': 'Month', 'level_2': 'Day'})
        pd_daily['Date'] = pd_daily.apply(lambda row:
                                          datetime(int(row['Year']), int(row['Month']), int(row['Day']), 1), axis=1)
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


def make_highcharts(actual_data, predictions):
    """Creates htmls with actual, prediction, and error plots with yearly aggregation.

    Args:
        actual_data (dictionary): Real data with structure {Aggregation: pandas DataFrame}
        predictions (dictionary): Predictions with structure {Year: {Aggregation: pandas DataFrame}}
    """

    # Convert real data to list of lists for plotting with highcharts
    actual = actual_data['Yearly'].reset_index()[['Date', 'ACT']].values.tolist()

    # Initialize list of years
    l_years = []

    # Create yearly plots
    for year in predictions:
        df = predictions[year]['Yearly']
        if year == '.Actual':
            year = 'Actual'
        l_years.append(year)

        # Define chart dimensions
        H = Highchart(width=1600, height=800)

        # Initialize options
        options = {
            'chart': {
                'type': 'column'
            },
            'title': {
                'text': 'Prediction with {} Weather'.format(year)
            },
            'subtitle': {
                'text': 'Click links above to change viewing year.'
            },
            'xAxis': {
                'type': 'category',
                'title': {
                    'text': 'Year'
                }
            },
            'yAxis': [{
                'labels': {
                    'format': '{value}',
                    'style': {
                        'color': 'Highcharts.getOptions().colors[2]'
                    }
                },
                'title': {
                    'text': 'Actual',
                    'style': {
                        'color': 'Highcharts.getOptions().colors[2]'
                    }
                },
                'opposite': True

            }, {
                'gridLineWidth': 0,
                'title': {
                    'text': 'Prediction',
                    'style': {
                        'color': 'Highcharts.getOptions().colors[0]'
                    }
                },
                'labels': {
                    'format': '{value}',
                    'style': {
                        'color': 'Highcharts.getOptions().colors[0]'
                    }
                }

            }, {
                'gridLineWidth': 0,
                'title': {
                    'text': 'Error',
                    'style': {
                        'color': 'Highcharts.getOptions().colors[1]'
                    }
                },
                'labels': {
                    'format': '{value}',
                    'style': {
                        'color': 'Highcharts.getOptions().colors[1]'
                    }
                },
                'opposite': True
            }],
            'tooltip': {
                'shared': True,

            },
            'legend': {
                'layout': 'vertical',
                'align': 'left',
                'x': 80,
                'verticalAlign': 'top',
                'y': 55,
                'floating': True,
                'backgroundColor': "(Highcharts.theme && Highcharts.theme.legendBackgroundColor) || '#FFFFFF'"
            },
        }

        # Convert pandas dataframe to lists of lists for plotting with highcharts
        error = df.reset_index()[['Date', 'Error']].values.tolist()
        prediction = df.reset_index()[['Date', 'Prediction']].values.tolist()

        # Plot with highcharts
        H.set_dict_options(options)
        H.add_data_set(actual, 'line', 'Actual', marker={'enabled': False})
        H.add_data_set(prediction, 'line', 'Prediction', marker={'enabled': False}, dashStyle='dash')
        H.add_data_set(error, 'column', 'Error')

        # Export plot
        filename = '/home/kimmmx/URD_Prediction_{}_Weather'.format(year)
        H.save_file(filename)

    # Open plot and replace beginning of file with links
    headstring = ""
    for year in l_years:
        filename = '/home/kimmmx/URD_Prediction_{}_Weather.html'.format(year)
        headstring += '<a href="{0}.html" style="color: #555; font-size: 36px">{1}</a> &ensp;'.format(filename[:-5], year)

    for year in l_years:
        filename = '/home/kimmmx/URD_Prediction_{}_Weather.html'.format(year)
        with open(filename, 'r') as f:
            content = f.read()
        with open(filename, 'w') as f:
            f.write(headstring)
            f.write(content)


def visualize_urd_highcharts(real_data, predictions, aggregations={'Yearly'}):
    """Creates html files to visualize real data and predictions for URD data using highcharts

        Args:

        Returns:

    """

    # Create dictionary of aggregation type to actual dataframe
    d_actual = aggregate_by_day_month_year(real_data, aggregations, 'Date1')

    # Create nested dictionary of applied weather year to aggregation type to prediction dataframe
    d_predictions = {}
    print("Aggregating data...")
    for prediction_year in tqdm(predictions):
        d_predictions[prediction_year] = aggregate_by_day_month_year(predictions[prediction_year],
                                                                     aggregations, 'Date1')

    make_highcharts(d_actual, d_predictions)

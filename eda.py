import matplotlib.pyplot as plt
import pandas as pd


def load_archive(path, with_datetime=True):
    """
    Loads a pandas dataframe from a .gz archive.
    Can load both the train and test datasets and bu_feat data.

    Adds datetime, year, month and day columns for convenience.
    :param path: path to the archive file
    :param with_datetime: whether the given dataframe contains datetime information.
        Set to True for train and test data, set to False for bu_feat data.
    :return:
        pd.DataFrame loaded from the archive
    """
    df = pd.read_csv(path, compression='gzip')

    if with_datetime:
        datetime = pd.to_datetime(df.day_id, yearfirst=True)

        df['datetime'] = datetime

        df['year'] = datetime.dt.year
        df['month'] = datetime.dt.month
        df['day'] = datetime.dt.day

        df = add_week_column(df)

    return df


def add_week_column(df):
    """
    Adds a week column to the dataframe using the isocalendar method on datetime data.
    :param df: input dataframe
    :return:
        pd.DataFrame with a week column ranging from 1 to 53
    """
    df['week'] = df.datetime.dt.isocalendar().week

    return df


def top_performing_dpt(df, year):
    """
    * For question 1-a with year 2016 *
    Gets the total turnover for the given year for each department.
    :param df: input dataframe with turnover data
    :param year: year to filter input data
    :return:
        total_turnover pandas series for each department
    """
    filtered_df = df[df.year == year]
    total_turnover = filtered_df.groupby(['dpt_num_department']).sum().turnover

    print(total_turnover.sort_values(ascending=False).head(1))

    return total_turnover


def top_performing_week(df, year, num_dpt):
    """
    * For question 1-b with year 2015 and department 88 *
    :param df: input dataframe with turnover data
    :param year: year to filter input data
    :param num_dpt: which department to analyze
    :return:
        total_turnover pandas series for each week
    """
    filtered_df = df[(df.year == year) & (df.dpt_num_department == num_dpt)]

    total_turnover = filtered_df.groupby('week').sum().turnover

    print(total_turnover.sort_values(ascending=False).head(5))

    return total_turnover


def top_performing_store(df, year):
    """
    * For question 1-c with year 2014 *
    :param df: input dataframe with turnover data
    :param year: year to filter input data
    :return:
        total_turnover pandas series for each store
    """
    filtered_df = df[df.year == year]

    total_turnover = filtered_df.groupby('but_num_business_unit').sum().turnover.sort_values(ascending=False)

    print(total_turnover.head(1))

    return total_turnover


def plot_weekly_evolution(df, num_dpt):
    """
    * For questions 1-d and 1-e*
    Plots the average weekly total turnover for the given department.

    Note that this weekly plot is not strictly accurate since it sums up all data with the same week number,
    but those numbers do not precisely correspond to the same weeks for each year.
    :param df: input dataframe with turnover data
    :param num_dpt: which department to analyze
    """
    weekly_data = df[df.dpt_num_department == num_dpt].groupby('week').sum().turnover

    plt.plot(weekly_data)

    plt.title(f'Evolution of weekly turnover data for department {num_dpt}')
    plt.xlabel('Week')
    plt.ylabel('Turnover')

    week_to_month = df.groupby('week').mean().month.round().to_list()
    list_loc = list()
    list_month = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                  'November', 'December']

    for month in range(1, 13):
        index = week_to_month.index(month)
        list_loc.append(index)

    plt.xticks(ticks=list_loc, labels=list_month)

    plt.show()

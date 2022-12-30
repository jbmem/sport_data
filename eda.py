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

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model, model_selection, ensemble, utils


def adding_dummies_dpt(df):
    """
    Replacing the num department column by dummy columns.
    This way, the model won't try to interpret the department numbers as numerical features.
    :param df: input dataframe
    :return: dataframe with dummy columns
    """
    df = df.join(pd.get_dummies(df.dpt_num_department, prefix='dpt'))

    return df


def add_long_lat(df, bu_feat):
    """
    Adds a longitude et latitude columns to the training features to replace the business unit number.
    :param df: input dataframe
    :param bu_feat: the business unit features dataframe
    :return: dataframe with added latitude and longitude columns
    """
    num_but = df.but_num_business_unit

    lat_dict = dict(zip(bu_feat.but_num_business_unit, bu_feat.but_latitude))
    long_dict = dict(zip(bu_feat.but_num_business_unit, bu_feat.but_longitude))

    lat_column = [lat_dict[but] for but in num_but]
    long_column = [long_dict[but] for but in num_but]

    df['but_latitude'] = pd.Series(lat_column)
    df['but_longitude'] = pd.Series(long_column)

    return df


def reduce_features(df):
    """
    Drops the columns not retained as features for the training of the model.
    :param df: input dataframe
    :return: dataframe with only the feature columns and the turnover column
    """
    return df.drop(['day_id', 'but_num_business_unit', 'dpt_num_department', 'datetime'], axis=1)


def prepare_for_training(df, bu_feat):
    """

    :param df:
    :param bu_feat:
    :return:
    """
    output = adding_dummies_dpt(df)
    output = add_long_lat(output, bu_feat)
    output = reduce_features(output)

    output = utils.shuffle(output)

    return output


def extract_x_y(df, without_y=False):
    """
    Extracts x (features) and y (labels) data from a dataframe
    :param df: input dataframe
    :param without_y: whether to also extract the labels
    :return:
        x: features
        y: (optional) labels
    """
    x = df.drop('turnover', axis=1).to_numpy()

    if without_y:
        return x
    else:
        y = df.turnover.to_numpy()
        return x, y


def train_test_regressor(train_df, bu_feat):
    """
    Train and evaluate linear regression methods
    :param train_df: train dataframe
    :param bu_feat: business unit features dataframe
    :return:
        x_train: the training features after scaling
        y_train: true labels
        y_prediction: predicted labels
    """
    train_df = prepare_for_training(train_df, bu_feat)

    x_train, y_train = extract_x_y(train_df)

    scaler_feat = StandardScaler()
    scaler_feat.fit(x_train)

    x_train = scaler_feat.transform(x_train)

    for alpha in [0., 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:

        regressor = linear_model.Ridge(alpha=alpha)
        regressor.fit(x_train, y_train)

        print(f'Ridge regression score for alpha {alpha}: {regressor.score(x_train, y_train)}')
        cv_scores = model_selection.cross_val_score(regressor, x_train, y_train)
        print(f'Cross validation score : {cv_scores}')
        print(f'Average cross validation score : {np.mean(cv_scores)}')
        print()

    regressor = linear_model.ElasticNet(alpha=1.)
    regressor.fit(x_train, y_train)

    y_pred = regressor.predict(x_train)

    return x_train, y_train, y_pred


def train_test_ensemble(train_df, bu_feat):
    """
    Train and evaluate ensemble methods
    :param train_df: train dataframe
    :param bu_feat: business unit features dataframe
    :return:
        x_train: the training features after scaling
        y_train: true labels
        y_prediction: predicted labels
    """
    train_df = prepare_for_training(train_df, bu_feat)

    x_train, y_train = extract_x_y(train_df)

    scaler_feat = StandardScaler()
    scaler_feat.fit(x_train)

    x_train = scaler_feat.transform(x_train)

    regressor = ensemble.GradientBoostingRegressor(n_estimators=100)
    regressor.fit(x_train, y_train)

    print(f'Ensemble regression score : {regressor.score(x_train, y_train)}')
    cv_scores = model_selection.cross_val_score(regressor, x_train, y_train)
    print(f'Cross validation score : {cv_scores}')
    print(f'Average cross validation score : {np.mean(cv_scores)}')
    print()

    y_prediction = regressor.predict(x_train)

    return x_train, y_train, y_prediction


if __name__ == '__main__':
    import eda

    train = eda.load_archive('test_data_scientist/train.csv.gz')
    bu_feat = eda.load_archive('test_data_scientist/bu_feat.csv.gz', with_datetime=False)

    x_t, y_t, y_p = train_test_regressor(train, bu_feat)

    print()

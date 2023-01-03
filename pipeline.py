import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor, nn, optim
from torch.utils.data import TensorDataset, DataLoader

import eda
from ml_model import TurnoverModel


def extract_series(series, seq_len):
    """
    Function to turn a time series into a batch of smaller sequences to be processed by the neural network model.
    :param series: numpy array of the turnover time series
    :param seq_len: length of the sequence, default to 10
    :return: features/labels pair
    """
    x, y = list(), list()

    scaler = MinMaxScaler()
    series = scaler.fit_transform(series.reshape(-1, 1))

    for t in range(len(series) - seq_len):
        x.append(series[t:t+seq_len])
        y.append(series[t+seq_len])

    x = np.array(x).reshape(-1, seq_len, 1)
    y = np.array(y)

    return x, y


def get_time_series(df, but, dpt, seq_len):
    """
    Gets the time series given a dataframe and a but/dpt key pair.
    Also normalizes the turnover data and splits into train/validation pair.
    Note that we do not shuffle the data before the train/val split, because it is sequential.
    :param df: input dataframe with turnover data
    :param but: business unit number
    :param dpt: department number
    :param seq_len: length of the smaller sequences
    :return:
        two features/labels pairs, one for training and one for validation
    """
    matching_data = df[(df.but_num_business_unit == but) & (df.dpt_num_department == dpt)]
    time_series = matching_data.sort_values(by='datetime').turnover.to_numpy()

    if time_series.shape[0] < seq_len:
        return None, None, None, None

    x, y = extract_series(time_series, seq_len)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle=False)

    return x_train, x_val, y_train, y_val


def get_all_time_series(df, seq_len):
    """
    Forms a dictionary with but/dpt keys
    :param df: dataframe with but and dpt columns
    :param seq_len: length of the moving window on the time series
    :return: dictionary with all the data grouped by but/dpt pair
    """
    group_obj = df.groupby(['but_num_business_unit', 'dpt_num_department'])

    dict_time_series = dict()

    for but, dpt in group_obj.groups.keys():
        x_t, x_v, y_t, y_v = get_time_series(df, but, dpt, seq_len)
        if x_t is not None:
            dict_time_series[(but, dpt)] = (x_t, x_v, y_t, y_v)

    return dict_time_series


def prepare_data(x_t, x_v, y_t, y_v):
    """
    Prepares the data using the DataLoader class.
    :param x_t: training features
    :param x_v: validation features
    :param y_t: training target
    :param y_v: validation target
    :return: pair of DataLoaders, one for training and one for validation
    """
    train_data, val_data = TensorDataset(Tensor(x_t), Tensor(y_t)), TensorDataset(Tensor(x_v), Tensor(y_v))
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    return train_loader, val_loader


def train_model(model, x_t, x_v, y_t, y_v):
    """
    Main training block of the LSTM based model.
    :param model: NN model based on the architecture defined in ml_model.py
    :param x_t: training features
    :param x_v: validation features
    :param y_t: training target
    :param y_v: validation target
    :return:
        model: trained model
        train_loss: MSE loss computed on the training data
        val_loss: MSE loss computed on the validation data
    """
    t_loader, val_loader = prepare_data(x_t, x_v, y_t, y_v)

    loss_fn = nn.MSELoss(reduction='sum')
    opt = optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 50

    # epoch training
    for t in range(n_epochs):
        for x_t, y_t in t_loader:
            y_pred = model(x_t)
            loss = loss_fn(y_pred.float(), y_t)

            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

    # compute final train and val loss
    with torch.no_grad():
        train_loss, val_loss = 0, 0

        for x_t, y_t in t_loader:
            train_pred = model(x_t)
            train_loss += loss_fn(train_pred.float(), y_t)

        for x_v, y_v in val_loader:
            val_pred = model(x_v)
            val_loss += loss_fn(val_pred.float(), y_v)

    return model, train_loss/len(t_loader), val_loss/len(val_loader)


def train_model_dict(dict_train, n_hidden=5, seq_len=10):
    """
    Forms a dictionary of trained models, one for each but/dpt key pair.
    :param dict_train: the dictionary of grouped data
    :param n_hidden: number of hidden neurons in the LSTM
    :param seq_len: length of each sequence sent to the model
    :return: the dictionary of trained models
    """
    dict_models = dict()

    for idx, (but, dpt) in enumerate(dict_train):
        x_t, x_v, y_t, y_v = dict_train[(but, dpt)]

        t_model = TurnoverModel(n_hidden=n_hidden, seq_len=seq_len)

        trained_model, t_loss, v_loss = train_model(t_model, x_t, x_v, y_t, y_v)
        dict_models[(but, dpt)] = (trained_model, t_loss, v_loss)

        print(f'Index {idx}, train loss is {t_loss} and val loss is {v_loss}')

    return dict_models


if __name__ == '__main__':

    train = eda.load_archive('test_data_scientist/train.csv.gz')
    d_train = get_all_time_series(train, 10)
    d_models = train_model_dict(d_train)

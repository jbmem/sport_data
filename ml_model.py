from torch import nn


class TurnoverModel(nn.Module):

    def __init__(self, n_hidden, seq_len, n_layers=1):
        """
        Creates a NN model composed of a LSTM layer followed by a Dense layer for output prediction
        :param n_hidden: number of hidden neurons in the LSTM layer
        :param seq_len: length of each sequence
        :param n_layers: number of LSTM layers
        """
        super(TurnoverModel, self).__init__()

        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size=1, hidden_size=self.n_hidden, num_layers=self.n_layers, dropout=0.2)
        self.dense = nn.Linear(in_features=n_hidden, out_features=1)

        self.hidden = None

    def forward(self, seq):
        """
        Forward pass of the model
        :param seq: input sequence to process
        :return: output prediction
        """
        if self.hidden is None:
            lstm_out, self.hidden = self.lstm(seq.view(self.seq_len, len(seq), -1))
        else:
            self.hidden = self.hidden[0].detach(), self.hidden[1].detach()
            lstm_out, self.hidden = self.lstm(seq.view(self.seq_len, len(seq), -1), self.hidden)

        last_time_step = lstm_out.view(self.seq_len, len(seq), self.n_hidden)[-1]
        y_prediction = self.dense(last_time_step)

        return y_prediction

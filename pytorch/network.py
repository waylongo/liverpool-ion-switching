import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_toolbelt import losses as L


class Seq2SeqCnnRnn(nn.Module):

    def __init__(self,
                 input_size,
                 seq_len,
                 hidden_size,
                 output_size,
                 num_layers=1,
                 bidirectional=False,
                 dropout=0.3,
                 kernel_size=5,
                 conv_layers=[64, 64],
                 hidden_layers=[64, 64]):

        super().__init__()
        self.input_size = input_size  # 1
        self.seq_len = seq_len  # 2500
        self.hidden_size = hidden_size  # 128
        self.num_layers = num_layers  # 2
        self.bidirectional = bidirectional  # True
        self.output_size = output_size  # 11
        self.kernel_size = kernel_size

        # CNN
        self.cov1 = nn.Conv1d(in_channels=input_size,
                              out_channels=conv_layers[0],
                              kernel_size=kernel_size,
                              padding=int((kernel_size - 1) / 2))
        self.cov2 = nn.Conv1d(in_channels=conv_layers[0],
                              out_channels=conv_layers[1],
                              kernel_size=kernel_size,
                              padding=int((kernel_size - 1) / 2))
        self.cov3 = nn.Conv1d(in_channels=conv_layers[1],
                              out_channels=conv_layers[2],
                              kernel_size=kernel_size,
                              padding=int((kernel_size - 1) / 2))
        # RNN
        self.rnn = nn.GRU(input_size=conv_layers[-1],
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=dropout)

        # Input Layer
        if hidden_layers and len(hidden_layers):  # [128, 64, 64]
            first_layer = nn.Linear(
                hidden_size * 2 if bidirectional else hidden_size,  # 128
                hidden_layers[0])

            # Hidden Layers
            self.hidden_layers = nn.ModuleList([first_layer] + [
                nn.Linear(hidden_layers[i], hidden_layers[i + 1])
                for i in range(len(hidden_layers) - 1)
            ])
            for layer in self.hidden_layers:
                nn.init.kaiming_normal_(layer.weight.data)

            self.intermediate_layer = nn.Linear(hidden_layers[-1],
                                                self.input_size)
            # output layers
            self.output_layer = nn.Linear(hidden_layers[-1], output_size)
            nn.init.kaiming_normal_(self.output_layer.weight.data)

        else:
            self.hidden_layers = []
            self.intermediate_layer = nn.Linear(
                hidden_size * 2 if bidirectional else hidden_siz,
                self.input_size)
            self.output_layer = nn.Linear(
                hidden_size * 2 if bidirectional else hidden_size, output_size)
            nn.init.kaiming_normal_(self.output_layer.weight.data)

        self.activation_fn = torch.relu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # cnn
        x = self.cov1(x)
        x = self.cov2(x)
        x = self.cov3(x)

        x = x.permute(0, 2, 1)

        # rnn
        outputs, hidden = self.rnn(x)
        x = self.dropout(self.activation_fn(outputs))

        # mlp
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
            x = self.dropout(x)

        x = self.output_layer(x)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_toolbelt import losses as L


class Wave_Block(nn.Module):

    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.dropout = nn.Dropout(0.2)
        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate * (kernel_size - 1)) / 2), dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate * (kernel_size - 1)) / 2), dilation=dilation_rate))
            self.convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x

        return self.dropout(res)


class WaveNet(nn.Module):

    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(1, 16, kernel_size=kernel_size, padding=int((kernel_size - 1) / 2))
        self.conv2d_2 = nn.Conv2d(16, 1, kernel_size=kernel_size, padding=int((kernel_size - 1) / 2))
        self.wave_block1 = Wave_Block(in_channels, 64, 8, kernel_size)
        self.wave_block2 = Wave_Block(64, 128, 4, kernel_size)

        self.fc = nn.Linear(128, 11)

    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = x.view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
        x = self.wave_block1(x)
        x = self.wave_block2(x)
        # x = self.wave_block3(x)

        x = x.permute(0, 2, 1)
        x = self.fc(x)

        return x

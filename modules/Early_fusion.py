import torch.nn as nn
import torch


class Early_fusion(nn.Module):

    def __init__(self, num_segments):
        super().__init__()
        self.lstm_visual = nn.LSTM(input_size=num_segments, hidden_size=num_segments, batch_first=True, bidirectional=False, num_layers=1)

    def forward(self, x):
        x_original = x

        b, t, c, h, w = x.size()
        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape(-1, w, t)

        x, _ = self.lstm_visual(x.float())

        x = x.reshape(b, c, h, w, t)
        x = x.permute(0,4,1,2,3)

        self.lstm_visual.flatten_parameters()
        x = torch.cat((x, x_original[:, x.size(1):, ...].contiguous()), dim=1)
        x = x.type(x_original.dtype) + x_original
        return x.mean(dim=1, keepdim=False)


class Late_fusion(nn.Module):

    def __init__(self, num_segments=2):
        super().__init__()
        self.lstm_visual = nn.LSTM(input_size=num_segments, hidden_size=num_segments, batch_first=True, bidirectional=False, num_layers=1)

    def forward(self, x):
        x_original = x

        b, t, c = x.size()
 
        x, _ = self.lstm_visual(x.float())

        self.lstm_visual.flatten_parameters()
        x = torch.cat((x, x_original[:, x.size(1):, ...].contiguous()), dim=1)
        x = x.type(x_original.dtype) + x_original
        return x.mean(dim=2, keepdim=False)
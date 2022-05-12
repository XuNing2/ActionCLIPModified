from torch import nn
from functools import partial
from timm.models.layers import DropPath, Mlp, PatchEmbed as TimmPatchEmbed

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size=None,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                           bias=bias, bidirectional=bidirectional)
    def forward(self, x):
        x,_ = self.rnn(x)
        return x



class VanillaSequencerBlock(nn.Module):
    def __init__(self, dim, hidden_size, mlp_ratio=3.0, rnn_layer=LSTM, mlp_layer=Mlp,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU,
                 num_layers=1, bidirectional=True, drop=0., drop_path=0.):
        super().__init__()
        channels_dim = int(mlp_ratio * dim)
        self.norm1 = norm_layer(dim)
        self.rnn_tokens = rnn_layer(dim, hidden_size, num_layers=num_layers, bidirectional=bidirectional)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim*2)
        self.mlp_channels = mlp_layer(dim*2, channels_dim, act_layer=act_layer, drop=drop,out_features =512)

    def forward(self, x):
        x = self.norm1(x)
        x = self.rnn_tokens(x)
        x = x + self.drop_path(x)

        x = self.norm2(x)
        x = self.mlp_channels(x)
        x = x + self.drop_path(x)
        # x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x
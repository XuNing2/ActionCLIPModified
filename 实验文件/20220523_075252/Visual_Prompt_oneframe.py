# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch
from torch import dropout, nn
from collections import OrderedDict
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from linformer_pytorch import MHAttention,Linformer,LinearAttentionHead
from modules.NextVLAD import NeXtVLAD
from modules.ChannelSELayer import ChannelSELayer
from modules.TCN import TemporalConvNet
from modules.Sequencer2DBlock import VanillaSequencerBlock as Sequencer

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


def trunc_normal_(x, mean=0., std=1.):
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)


class TAggregate(nn.Module):
    def __init__(self, clip_length=None, embed_dim=2048, n_layers=6):
        super(TAggregate, self).__init__()
        self.clip_length = clip_length
        drop_rate = 0.
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers, norm=nn.LayerNorm(
            embed_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, clip_length + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        with torch.no_grad():
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        nvids = x.shape[0]

        cls_tokens = self.cls_token.expand(nvids, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x.transpose_(1, 0)
        o = self.transformer_enc(x)

        return o[0]


class TemporalTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))

class NewNet(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        x_original = x
        b, t, c = x.size()
        seq_length = t
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        x = x + frame_position_embeddings

        x = x.permute(1, 0, 2)  # NLD -> LND
        return self.resblocks(x)


class visual_prompt(nn.Module):
    def __init__(self, sim_head, clip_state_dict, T,batchsize):
        super().__init__()
        self.sim_header = sim_head
        self.T = T
        # assert sim_head in ["meanP", "LSTM", "Transf", "Conv_1D", "Transf_cls","new_net","RNN","BiLSTM"]

        if self.sim_header == "LSTM" or self.sim_header == "Transf" or self.sim_header == "Transf_cls" or self.sim_header == "Conv_1D" or self.sim_header == "new_net"\
            or self.sim_header == "RNN" or self.sim_header == "BiLSTM" or self.sim_header == "NextVLAD" or self.sim_header =="GRU"\
                or self.sim_header=="TCN" or self.sim_header=="Sequencer":
            self.embed_dim = clip_state_dict["text_projection"].shape[1]
            context_length = clip_state_dict["positional_embedding"].shape[0]
            vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
            transformer_width = clip_state_dict["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64
            self.h = transformer_heads

            transformer_layers = len(
                set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

            self.frame_position_embeddings = nn.Embedding(context_length, self.embed_dim)
        if self.sim_header=="Sequencer":
            self.Sequencer = Sequencer(dim=512,hidden_size=512)
        if self.sim_header =="TCN":
            self.TCN = TemporalConvNet(num_inputs=8,num_channels=[self.embed_dim,self.embed_dim,8])
        if self.sim_header == "NextVLAD":
            self.nextVLAD = NeXtVLAD(feature_size=512,output_size=512,dropout_prob=0.0)
        if self.sim_header == "GRU":
            self.lstm_visual = nn.GRU(input_size=self.embed_dim, hidden_size=self.embed_dim,
                                       batch_first=True, bidirectional=False, num_layers=2,dropout=0)
        if self.sim_header == "ChannelSELayer":
            self.ChannelSELayer = ChannelSELayer(num_channels=16)
        if self.sim_header == "Transf" :
            layers = 6
            self.transformer = TemporalTransformer(width=self.embed_dim, layers=layers, heads=transformer_heads)
            print('layer='+str(layers))
        if self.sim_header == "new_net":
            self.new_net = Linformer(
                input_size=self.embed_dim, # Dimension 1 of the input
                channels=8, # Dimension 2 of the input
                dim_d=128, # The inner dimension of the attention heads
                dim_k=128, # The second dimension of the P_bar matrix from the paper
                dim_ff=128, # Dimension in the feed forward network
                dropout_ff=0.15, # Dropout for feed forward network
                nhead=transformer_heads, # Number of attention heads
                depth=6, # How many times to run the model
                dropout=0, # How much dropout to apply to P_bar after softmax
                activation="gelu", # What activation to use. Currently, only gelu and relu supported, and only on ff network.
                checkpoint_level="C2", # What checkpoint level to use. For more information, see below.
            )
            # pass
        if self.sim_header == "LSTM":
            self.lstm_visual = nn.LSTM(input_size=self.embed_dim, hidden_size=self.embed_dim,
                                       batch_first=True, bidirectional=False, num_layers=1)
        if self.sim_header == "BiLSTM":
            self.lstm_visual = nn.LSTM(input_size=self.embed_dim, hidden_size=self.embed_dim,
                                       batch_first=True, bidirectional=True, num_layers=1)

        if self.sim_header == "RNN":
            self.rnn_visual = nn.RNN(input_size=self.embed_dim, hidden_size=self.embed_dim,
                                       batch_first=True, bidirectional=False, num_layers=1)


        self.apply(self.init_weights)

        if self.sim_header == "Transf_cls":
            self.transformer = TAggregate(clip_length=self.T, embed_dim=self.embed_dim, n_layers=6)

        if self.sim_header == 'Conv_1D' :
            self.shift = nn.Conv1d(self.embed_dim, self.embed_dim, 3, padding=1, groups=self.embed_dim, bias=False)
            weight = torch.zeros(self.embed_dim, 1, 3)
            weight[:self.embed_dim // 4, 0, 0] = 1.0
            weight[self.embed_dim // 4:self.embed_dim // 4 + self.embed_dim // 2, 0, 1] = 1.0
            weight[-self.embed_dim // 4:, 0, 2] = 1.0
            self.shift.weight = nn.Parameter(weight)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        b, t, c = x.size()
        x = x.contiguous()
        if self.sim_header == "meanP":
            pass
        elif self.sim_header == 'Conv_1D':
            x_original = x
            x = x.view(-1, c, t)
            x = self.shift(x.float())
            x = x.permute(0, 2, 1)
            x = x.type(x_original.dtype) + x_original

        elif self.sim_header == "Transf":
            x_original = x
            seq_length = t
            position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            x = x + frame_position_embeddings

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = x.type(x_original.dtype) + x_original
        elif self.sim_header == "NextVLAD":
            x_original = x
            y = self.nextVLAD(x.float())
            x= y
            return x.type(x_original.dtype)
        elif self.sim_header == "ChannelSELayer":
            x_original = x
            y = self.ChannelSELayer(x.float())
            x= y
            return x.type(x_original.dtype)
        elif self.sim_header == "TCN":
            x_original = x
            y = self.TCN(x.float())
            x= y
            x=x.type(x_original.dtype) + x_original
        elif self.sim_header == "LSTM" or self.sim_header == "BiLSTM" or self.sim_header=="GRU":
            x_original = x
            x, _ = self.lstm_visual(x.float())
            self.lstm_visual.flatten_parameters()
            x = torch.cat((x, x_original[:, x.size(1):, ...].contiguous()), dim=1)
            x = x.type(x_original.dtype) + x_original
        elif self.sim_header == "RNN":
            x_original = x
            x, _ = self.rnn_visual(x.float())
            self.rnn_visual.flatten_parameters()
            x = torch.cat((x, x_original[:, x.size(1):, ...].contiguous()), dim=1)
            x = x.type(x_original.dtype) + x_original
        elif self.sim_header == "Transf_cls":
            x_original = x
            return self.transformer(x).type(x_original.dtype)
        elif self.sim_header == "new_net":
            
            x_original = x
            x=x.permute(0,2,1)
            y=self.new_net(x.float())
            x=y
            x=x.permute(0,2,1)
            x = x.type(x_original.dtype) + x_original
        elif self.sim_header == "Sequencer":
            x_original = x
            y=self.Sequencer(x.float())
            x=y
            x = x.type(x_original.dtype) + x_original

        else:
            raise ValueError('Unknown optimizer: {}'.format(self.sim_header))
        return x.mean(dim=1, keepdim=False)

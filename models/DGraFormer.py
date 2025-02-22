__all__ = ['DGraFormer']

import os
# Cell
from typing import Optional

import math
import numpy as np
import pandas as pd
import torch
import torch.fft as fft
from einops import rearrange, reduce, repeat
from torch import nn

from layers.DGraFormer_framework import DGraFormer_framework


class Model(nn.Module):
    def __init__(self, configs, d_k: Optional[int] = None, d_v: Optional[int] = None, norm: str = 'BatchNorm',
                 attn_dropout: float = 0., act: str = "gelu", res_attention: bool = True, pre_norm: bool = False,
                 store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True, pretrain_head: bool = False,
                 head_type='flatten', verbose: bool = False, **kwargs):
        super().__init__()

        # load parameters
        n_vars = configs.n_vars
        context_window = configs.seq_len
        target_window = configs.pred_len

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        predictor_dropout = configs.predictor_dropout

        patch_len = configs.patch_len
        stride = configs.stride

        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        device = configs.device
        d_graph = configs.d_graph
        d_gcn = configs.d_gcn
        w_ratio = configs.w_ratio
        mp_layers = configs.mp_layers

        root_path = configs.root_path
        data_path = configs.data_path
        data = configs.data
        df_raw = pd.read_csv(os.path.join(root_path, data_path))
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        if data == 'ETTh1' or 'ETTh2':
            train_data = df_data[0:12 * 30 * 24]
        elif data == 'ETTm1' or 'ETTm2':
            train_data = df_data[0:12 * 30 * 24 * 4]
        else:
            num_train = int(len(df_raw) * 0.7)
            train_data = df_data[0:num_train]
        cossim_matrix = overall_Matrix(train_data, device)

        # model
        self.model = DGraFormer_framework(cossim_matrix=cossim_matrix, n_vars=n_vars, context_window=context_window,
                                         target_window=target_window, patch_len=patch_len, stride=stride, device=device,
                                         d_graph=d_graph, d_gcn=d_gcn, w_ratio=w_ratio, mp_layers=mp_layers,
                                         n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v,
                                         d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout, act=act,
                                         res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn, pe=pe,
                                         learn_pe=learn_pe, predictor_dropout=predictor_dropout,
                                         pretrain_head=pretrain_head, head_type=head_type, revin=revin, affine=affine,
                                         subtract_last=subtract_last, verbose=verbose, **kwargs)

    def forward(self, x, time_index, current_epoch):  # x: [bs x seq_len x nvars]

        x = x.permute(0, 2, 1)  # x: [bs x nvars x seq_len]
        x = self.model(x, time_index, current_epoch)
        x = x.permute(0, 2, 1)  # x: [bs x seq_len x nvars]
        return x


def overall_Matrix(data, device, pred_len=0, k=3, low_freq=1):
    if 'date' in data.columns:
        data = data.drop(columns=['date'])
    data = data.dropna()

    variables = data.to_numpy().T
    variables_tensor = torch.tensor(variables, dtype=torch.float32, device=device)

    variables_tensor = variables_tensor.unsqueeze(0)

    fourier_layer = FourierLayer(pred_len=pred_len, k=k, low_freq=low_freq).to(device)

    processed_tensor, _ = fourier_layer(variables_tensor)

    processed_tensor = processed_tensor.squeeze(0)

    def cosine_similarity_matrix(variables_tensor):
        norms = torch.norm(variables_tensor, dim=1, keepdim=True)
        normalized = variables_tensor / norms
        similarity_matrix = torch.mm(normalized, normalized.T)

        return similarity_matrix

    similarity_tensor = cosine_similarity_matrix(processed_tensor)

    return similarity_tensor


class FourierLayer(nn.Module):

    def __init__(self, pred_len, k=None, low_freq=1, output_attention=False):
        super().__init__()
        self.pred_len = pred_len
        self.k = k
        self.low_freq = low_freq
        self.output_attention = output_attention

    def forward(self, x):
        """x: (b, t, d)"""
        if self.output_attention:
            return self.dft_forward(x)

        b, t, d = x.shape
        x_freq = fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2))
        f = f.to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)

        return self.extrapolate(x_freq, f, t), None

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t_val = rearrange(torch.arange(t + self.pred_len, dtype=torch.float),
                          't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs() / t, 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')

        x_time = amp * torch.cos(2 * math.pi * f * t_val + phase)

        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        values, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)))
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]

        return x_freq, index_tuple

    def dft_forward(self, x):
        T = x.size(1)

        dft_mat = fft.fft(torch.eye(T))
        i, j = torch.meshgrid(torch.arange(self.pred_len + T), torch.arange(T))
        omega = np.exp(2 * math.pi * 1j / T)
        idft_mat = (np.power(omega, i * j) / T).cfloat()

        x_freq = torch.einsum('ft,btd->bfd', [dft_mat, x.cfloat()])

        if T % 2 == 0:
            x_freq = x_freq[:, self.low_freq:T // 2]
        else:
            x_freq = x_freq[:, self.low_freq:T // 2 + 1]

        _, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        indices = indices + self.low_freq
        indices = torch.cat([indices, -indices], dim=1)

        dft_mat = repeat(dft_mat, 'f t -> b f t d', b=x.shape[0], d=x.shape[-1])
        idft_mat = repeat(idft_mat, 't f -> b t f d', b=x.shape[0], d=x.shape[-1])

        mesh_a, mesh_b = torch.meshgrid(torch.arange(x.size(0)), torch.arange(x.size(2)))

        dft_mask = torch.zeros_like(dft_mat)
        dft_mask[mesh_a, indices, :, mesh_b] = 1
        dft_mat = dft_mat * dft_mask

        idft_mask = torch.zeros_like(idft_mat)
        idft_mask[mesh_a, :, indices, mesh_b] = 1
        idft_mat = idft_mat * idft_mask

        attn = torch.einsum('bofd,bftd->botd', [idft_mat, dft_mat]).real
        return torch.einsum('botd,btd->bod', [attn, x]), rearrange(attn, 'b o t d -> b d o t')

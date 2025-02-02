__all__ = ['DGraFormer_backbone']

# Cell
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

# from collections import OrderedDict
from layers.DGraFormer_layers import *
from layers.RevIN import RevIN


# Cell
class DGraFormer_backbone(nn.Module):
    def __init__(self, c_in: int, context_window: int, target_window: int, patch_len: int, stride: int,
                 device: str = 'cpu', max_seq_len: Optional[int] = 1024, d_graph=30, d_gcn=1, mask=0.5,
                 mp_layers: int = 2, n_layers: int = 3, d_model=128, n_heads=16, d_k: Optional[int] = None,
                 d_v: Optional[int] = None,
                 d_ff: int = 256, norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True, fc_dropout: float = 0., head_dropout=0, padding_patch=None,
                 pretrain_head: bool = False, head_type='flatten', individual=False, revin=True, affine=True,
                 subtract_last=False,
                 verbose: bool = False, **kwargs):

        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int(context_window / stride)
        print(f"patch_num: {patch_num}")
        print(f"patch_len: {patch_len}")
        # Backbone
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                    n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                    attn_dropout=attn_dropout, dropout=dropout, act=act,
                                    key_padding_mask=key_padding_mask, padding_var=padding_var,
                                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                    store_attn=store_attn,
                                    pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.d_gcn = d_gcn
        self.head = Flatten_Head(self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)

        self.device = device
        self.d_graph = d_graph
        self.mask = mask
        self.mp_layers = mp_layers

        self.gc = graph_constructor(self.n_vars, self.d_graph, self.device, mask=self.mask)
        self.gcn1 = DGCN(d_gcn=self.d_gcn, mp_layers=self.mp_layers)
        self.gcn2 = DGCN(d_gcn=self.d_gcn, mp_layers=self.mp_layers)


    def forward(self, z, time_index, current_epoch):  # z: [bs x nvars x seq_len] # time_index: [bs x timeinfo]
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)

        indices = time_index // 24
        indices = indices % 7

        adj = self.gc(indices, current_epoch)
        z = self.gcn1(z, adj)

        # do patching
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x patch_len x patch_num]

        # model
        z = self.backbone(z)  # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)  # z: [bs x nvars x target_window]

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                             nn.Conv1d(head_nf, vars, 1)
                             )


class DGCN(nn.Module):
    def __init__(self, d_gcn, mp_layers, c_in=1, c_out=1, propbeta=0.05):
        super(DGCN, self).__init__()
        self.nconv = nconv()
        self.d_gcn = d_gcn
        if d_gcn != 1:
            self.startmlp = nn.Linear(c_in, d_gcn)
            self.mlp = nn.Linear((mp_layers + 1) * d_gcn, c_out)
        else:
            self.mlp = nn.Linear((mp_layers + 1) * c_in, c_out)

        self.mp_layers = mp_layers
        self.propbeta = propbeta

    def forward(self, x, adj):
        a = adj.permute(0, 2, 3, 1)
        x = x.unsqueeze(-1)
        if self.d_gcn != 1:
            h = self.startmlp(x)
        else:
            h = x
        out = [h]
        for i in range(self.mp_layers):
            h = self.propbeta * x + (1 - self.propbeta) * self.nconv(h, a)
            out.append(h)

        h = torch.cat(out, dim=-1)

        h = self.mlp(h)
        h = h.squeeze(-1)  # 移除最后一个维度
        return h


class graph_constructor(nn.Module):
    def __init__(self, n_vars, d_graph, device, alpha=0.9, num_adj_matrices=7, mask=0.5):
        super(graph_constructor, self).__init__()
        self.n_vars = n_vars

        # 读取 CSV 文件，假设文件名为 'cosine_similarity_matrix.csv'
        file_path = './layers/ETTh1_top3_cosine_similarity_matrix_after_fourier01.csv'
        data = pd.read_csv(file_path, index_col=0)

        # 将 DataFrame 转换为 NumPy 数组
        similarity_matrix = data.to_numpy()
        self.init_adj_matrix = torch.tensor(similarity_matrix, device=device, dtype=torch.float32)

        # 预生成节点嵌入矩阵
        self.emb_list1 = nn.ParameterList(
            [nn.Parameter(torch.randn(n_vars, d_graph, device=device)) for _ in range(num_adj_matrices)])
        self.emb_list2 = nn.ParameterList(
            [nn.Parameter(torch.randn(n_vars, d_graph, device=device)) for _ in range(num_adj_matrices)])

        # 用于生成不同的邻接矩阵
        self.lin1 = nn.ModuleList([nn.Linear(d_graph, d_graph) for _ in range(num_adj_matrices)])
        self.lin2 = nn.ModuleList([nn.Linear(d_graph, d_graph) for _ in range(num_adj_matrices)])
        self.lin1.to(device)
        self.lin2.to(device)

        self.device = device
        self.alpha = alpha
        self.num_adj_matrices = num_adj_matrices
        self.mask = mask

    def forward(self, time_indices, current_epoch):
        adjs = []
        num_elements = self.n_vars * self.n_vars
        for i in range(self.num_adj_matrices):
            nodevec1 = torch.tanh(self.lin1[i](self.emb_list1[i]))
            nodevec2 = torch.tanh(self.lin2[i](self.emb_list2[i]))

            prop = min(current_epoch / 5, self.alpha)  # 限制 part 在 [0, 1]
            a = (1 - prop) * self.init_adj_matrix + prop * torch.mm(nodevec1, nodevec2.transpose(1, 0))
            adj = F.relu(torch.tanh(a))

            # 将对角线设置为0
            adj = adj - torch.diag(torch.diag(adj))

            values, indices = torch.topk(adj.reshape(-1), int(num_elements * self.mask), largest=True)

            # 创建一个零的mask矩阵
            mask = torch.zeros_like(adj.reshape(-1), device=adj.device)
            mask[indices] = 1  # 选择 topk 的元素，其他位置为 0

            # 使用 mask 来筛选邻接矩阵
            adj = mask.view(adj.size(0), adj.size(1)).view_as(adj) * adj

            # 将对角线重新设置为 1
            adj = adj + torch.eye(adj.size(0)).to(adj.device)

            d = adj.sum(1)
            adj = adj / d.view(-1, 1)

            adjs.append(adj)

        oadj = torch.stack(adjs)
        dadj = oadj[time_indices]
        return dadj


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('bnsc,bnms->bmsc', (x, A))
        return x.contiguous()


class Flatten_Head(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]

        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)

        return x


class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder1 = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                   attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                                   store_attn=store_attn)
        self.encoder2 = TSTEncoder(q_len / 2, d_model * 2, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                   attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                                   store_attn=store_attn)
        self.encoder3 = TSTEncoder(q_len / 4, d_model * 4, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                   attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                                   store_attn=store_attn)

    def forward(self, x) -> Tensor:  # x: [bs x nvars x patch_len x patch_num]

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # u: [bs * nvars x patch_num x d_model]

        u = self.dropout(u + self.W_pos)  # u: [bs * nvars x patch_num x d_model]

        # Encoder
        u = self.encoder1(u)  # z: [bs * nvars x patch_num x d_model]

        b, n, d = u.shape
        u = u.reshape(b, n // 2, 2 * d)

        # Encoder
        u = self.encoder2(u)  # z: [bs * nvars x patch_num x d_model]

        b, n, d = u.shape
        u = u.reshape(b, n // 2, 2 * d)
        # Encoder
        u = self.encoder3(u)  # z: [bs * nvars x patch_num x d_model]

        u = torch.reshape(u, (-1, n_vars, u.shape[-2], u.shape[-1]))  # z: [bs x nvars x patch_num x d_model]
        u = u.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x patch_num]

        return u

    # Cell


class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList(
            [TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                             attn_dropout=attn_dropout, dropout=dropout,
                             activation=activation, res_attention=res_attention,
                             pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask,
                                                         attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False,
                 pre_norm=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout,
                                             proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        # Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask,
                                                attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        # Add & Norm
        src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        # Position-wise Feed-Forward
        src2 = self.ff(src)
        # Add & Norm
        src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0.,
                 qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout,
                                                   res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,
                                                                         2)  # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3,
                                                                       1)  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)  # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1,
                                                          self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights

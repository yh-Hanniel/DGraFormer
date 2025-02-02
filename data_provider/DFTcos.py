import math
import numpy as np
import pandas as pd
import torch
import torch.fft as fft
import torch.nn as nn
from einops import rearrange, reduce, repeat

# 读取ETTh1.csv文件
file_path = 'ETTh1.csv'
data = pd.read_csv(file_path)

# 假设时间戳列名为 'date'，如果存在则移除
if 'date' in data.columns:
    data = data.drop(columns=['date'])

# 检查是否有缺失值，移除或填充缺失值（此处选择移除含缺失值的行）
data = data.dropna()

total_days = 12 * 30  # 前 12 个月的数据（每月 30 天）
points_per_day = 24   # 每天 24 个时间点（小时级别）
total_points = total_days * points_per_day  # 总数据点数

data = data.head(total_points)  # 仅保留前 total_points 个数据点

# 转换为numpy数组，列为变量（每行表示一个变量）
variables = data.to_numpy().T  # 转置后每行表示一个变量


class FourierLayer(nn.Module):

    def __init__(self, pred_len, k=None, low_freq=1, output_attention=False):
        super().__init__()
        # self.d_model = d_model
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


# 将 variables 转换为 PyTorch 张量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
variables_tensor = torch.tensor(variables, dtype=torch.float32, device=device)

# 添加批量维度
variables_tensor = variables_tensor.unsqueeze(0)  # 形状变为 (1, 321, 18620)

# 初始化 FourierLayer
fourier_layer = FourierLayer(pred_len=0, k=3).to(device)

# 使用 FourierLayer 处理数据
processed_tensor, _ = fourier_layer(variables_tensor)

# 移除批量维度
processed_tensor = processed_tensor.squeeze(0)  # 形状恢复为 (321, 18620)

# 转换为 NumPy 格式
processed_numpy = processed_tensor.cpu().numpy()


# 计算余弦相似度矩阵
def cosine_similarity_matrix(variables):
    # 计算每个变量（行）的L2范数
    norms = np.linalg.norm(variables, axis=1, keepdims=True)

    # 归一化变量
    normalized = variables / norms

    # 计算余弦相似度矩阵
    similarity_matrix = np.dot(normalized, normalized.T)

    return similarity_matrix


# 获取余弦相似度矩阵
similarity_matrix = cosine_similarity_matrix(processed_numpy)

# 只保留大于 0.7 的相似度值，其余设置为 0
# similarity_matrix[similarity_matrix <= 0.0] = 0

# 转换为 DataFrame，添加行列标签
similarity_df = pd.DataFrame(similarity_matrix, index=data.columns, columns=data.columns)

# 打印相似度矩阵
print("余弦相似度矩阵：")
print(similarity_df)

# 保存结果为 CSV 文件（可选）
similarity_df.to_csv('ETTh1_top3_cosine_similarity_matrix_after_fourier02.csv', index=True)

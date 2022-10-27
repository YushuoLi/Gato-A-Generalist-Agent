import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from Gato_models.config import GatoConfig
from torch.distributions.uniform import Uniform
from typing import Dict, Any, Union

from Gato_models.nets import ResidualBlock_V2

def _randomized_positions(from_v, to_v, device):
    # 随机选择某个位置的向量
    assert from_v.shape == to_v.shape
    min_val = torch.zeros(from_v.shape, dtype=torch.float32).to(device)
    max_val = torch.ones(to_v.shape, dtype=torch.float32).to(device)
    uni_dis = Uniform(min_val, max_val)
    gap = (to_v - from_v).clone().detach()
    pos = uni_dis.sample() * gap
    return pos.type(torch.int32)


def _rounded_mean_positions(from_v, to_v):
    mean = from_v + to_v
    pos = mean.type(torch.float32)
    pos = torch.round(pos / 2).type(torch.int32)
    return pos


def _broadcast(row_pos, col_pos, row_ones, col_ones):

    # broadcast (5,) to (20,) with column-axis  (height/patch_size,) to (height*width/patch_size^2,)
    row_pos = torch.unsqueeze(row_pos, 1).type(torch.float32)  # (5,1)
    col_ones = col_ones.type(torch.float32)  # @:"addmm_cuda" 不支持整数运算
    row_pos = row_pos @ col_ones  # (5,1) * (1,4)
    row_pos = row_pos.view(-1,)
    row_pos = row_pos.type(torch.int32).clone().detach().requires_grad_(False)

    # broadcast (4,) to (20,) with row-axis
    col_pos = torch.unsqueeze(col_pos, 0).type(torch.float32)  # (1,4)
    row_ones = row_ones.type(torch.float32)
    col_pos = row_ones @ col_pos  # (5,1) * (1,4)
    col_pos = col_pos.view(-1,)
    col_pos = col_pos.type(torch.int32).clone().detach().requires_grad_(False)

    return row_pos, col_pos


class PatchPositionEncoding(nn.Module):

    def __init__(self,
                 config: Union[GatoConfig, Dict[str, Any]],
                 mode='training'):
        """
        Appendix C.3. Position Encodings
        """
        super(PatchPositionEncoding, self).__init__()

        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config

        self.device = self.config.device

        assert config.img_height % self.config.img_patch_size == 0, 'Height must be divided by patch size with no remainders'
        assert config.img_width % self.config.img_patch_size == 0, 'Width must be divided by patch size with no remainders'

        embedding_dim = self.config.layer_width
        self.embedding_dim = embedding_dim
        self.discretize_depth = self.config.discretize_depth
        self.height = config.img_height
        self.width = config.img_width
        self.patch_size = self.config.img_patch_size

        self.rows_from = self.rows_to = self.cols_from = self.cols_to = None
        self.row_embedding = self.col_embedding = None
        self.row_train_pos = self.col_train_pos = None
        self.row_eval_pos = self.col_eval_pos = None

        self.mode = mode  # 训练 or 评估
        self.build()

    def _discretize(self, pos):
        return round(pos * self.discretize_depth)

    def _discretize_interval(self, axis_num):
        axis_from = []
        axis_to = []
        for index in range(axis_num // self.patch_size):
            from_pos = index * self.patch_size / axis_num
            to_pos = (index + 1) * self.patch_size / axis_num
            axis_from.append(self._discretize(from_pos))  # 映射到图片词典大小 discretize_depth：128
            axis_to.append(self._discretize(to_pos))
        return axis_from, axis_to

    def build(self):
        # Appendix C.3. Position Encodings; Figure 15 | Patch position encodings.
        rows_from, rows_to = self._discretize_interval(self.height)
        cols_from, cols_to = self._discretize_interval(self.width)

        self.rows_from = torch.tensor(rows_from, dtype=torch.int32).to(self.device)
        self.rows_to = torch.tensor(rows_to, dtype=torch.int32).to(self.device)
        self.cols_from = torch.tensor(cols_from, dtype=torch.int32).to(self.device)
        self.cols_to = torch.tensor(cols_to, dtype=torch.int32).to(self.device)

        # 由位置（0，discretize_depth）映射到嵌入空间
        self.row_embedding = nn.Embedding(self.discretize_depth, self.embedding_dim)
        self.col_embedding = nn.Embedding(self.discretize_depth, self.embedding_dim)

        row_ones = torch.ones((self.height // self.patch_size, 1), dtype=torch.int32).to(self.device)
        col_ones = torch.ones((1, self.width // self.patch_size), dtype=torch.int32).to(self.device)

        # 训练过程均匀随机采样 元素为每一个 patch 的对应位置
        self.row_train_pos, self.col_train_pos = _broadcast(self.rows_from +
                                                            _randomized_positions(self.rows_from, self.rows_to, self.device),
                                                            self.cols_from +
                                                            _randomized_positions(self.cols_from, self.cols_to, self.device),
                                                            row_ones, col_ones)
        # 评估过程取平均
        self.row_eval_pos, self.col_eval_pos = _broadcast(_rounded_mean_positions(self.rows_from, self.rows_to),
                                                          _rounded_mean_positions(self.cols_from, self.cols_to),
                                                          row_ones, col_ones)
        self.built = True

    def forward(self, inputs):
        # Appendix C.3. Position Encodings
        row_pos, col_pos = (
            (self.row_train_pos, self.col_train_pos) if self.mode == 'training' else (self.row_eval_pos, self.col_eval_pos)
        )
        # 从 embedding table 分别获得行列的 position encoding 并添加到 token embedding(input: (B, T, S, E)) S 为图片patch长度
        # broadcast (S,E) to (B, S, E)
        return inputs + self.row_embedding(row_pos) + self.col_embedding(col_pos)  # broadcast


class ResidualEmbedding(nn.Module):

    def __init__(self, config: Union[GatoConfig, Dict[str, Any]]):
        """
        Appendix C.2. Embedding Function
        """
        super(ResidualEmbedding, self).__init__()

        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config
        assert config.img_height % self.config.img_patch_size == 0, 'Height must be divided by patch size with no remainders'
        assert config.img_width % self.config.img_patch_size == 0, 'Width must be divided by patch size with no remainders'

        embedding_dim = self.config.layer_width
        self.embedding_dim = embedding_dim
        self.height = config.img_height
        self.width = config.img_width
        self.patch_size = self.config.img_patch_size
        self.num_group_norm_groups = self.config.num_group_norm_groups

        self.rows = self.height // self.patch_size
        self.cols = self.width // self.patch_size

        self.resnet_block_v2 = ResidualBlock_V2(config.in_channels, config.out_channels, config.stride, self.num_group_norm_groups)
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.out_channels * self.patch_size ** 2, embedding_dim)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        embedding_tokens = []
        for i in range(self.rows):
            for j in range(self.cols):
                patch = x[..., self.patch_size * i: self.patch_size * (i+1),
                      self.patch_size * j: self.patch_size * (j+1)]  # (B, C, _, _)
                embed = self.resnet_block_v2(patch)
                embed = self.linear(embed)
                embed = torch.unsqueeze(embed, 1)  # (B, emb) to (B, 1, emb)
                embedding_tokens.append(embed)

        embedding_tokens = torch.cat(embedding_tokens, 1)  # (B, len, emb)
        return embedding_tokens



class LocalPositionEncoding(nn.Module):

    def __init__(self, config: Union[GatoConfig, Dict[str, Any]]):
        """
        Appendix C.3. Position Encodings > Local Observation Position Encodings
        """
        super(LocalPositionEncoding, self).__init__()

        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config

        self.device = self.config.device

        self.embedding = nn.Embedding(self.config.local_position_encoding_size, self.config.layer_width)

    def forward(self, inputs, sym_index):
        # inputs:(B, T, (z,|,a)) (B, T, S+A+1, emd), sym_index: | 所在的位置下标
        input_size = inputs.shape[-2]
        pos_ob = torch.arange(0, sym_index, dtype=torch.int32).to(self.device)  # 观测向量的位置 (S,)
        pos_ac = torch.ones(input_size - sym_index - 1,
                            dtype=torch.int32).to(self.device) * (self.config.local_position_encoding_size-1)  # 动作向量的位置 (A,)
        emd_ob = self.embedding(pos_ob)  # (S, emd)
        emd_ac = self.embedding(pos_ac)  # (A, emd)
        emd_sep = torch.zeros((1, emd_ob.shape[-1])).to(self.device)  # (1, emd)
        pos_emd = torch.cat((emd_ob, emd_sep, emd_ac), dim=0)  # (S+1+A, emd)
        return inputs + pos_emd  # broadcast (S+A+1, emd) -> (B, T, S+A+1, emd)


class DiscreteEmbedding(nn.Module):
    """
    连续的观测和动作已经经过离散化过程了，因此输入全部是离散的
    """
    def __init__(self, config: Union[GatoConfig, Dict[str, Any]]):
        super(DiscreteEmbedding, self).__init__()

        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config

        # Appendix C.1. Transformer Hyperparameters
        # Shared Embedding
        self.embedding = nn.Embedding(self.config.embedding_input_size,
                                      self.config.layer_width,
                                      )

    def forward(self, inputs):
        return self.embedding(inputs)



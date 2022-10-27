import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Gato_models.config import GatoConfig
from typing import Union, Dict, Any


def mu_law_encode(x, mu=100, m=256):
    # Appendix B. Agent Data Tokenization Details
    # 连续观测序列进行 mu-law 压缩
    sign = torch.sign(x)
    numerator = torch.log(torch.abs(x) * mu + 1.0)
    denominator = math.log(m * mu + 1.0)
    return (numerator / denominator) * sign


def tokenize_obs_values(x, mu=100, m=256, bins=1024, obs_continuous=True, shift=32000):
    # Appendix B. Agent Data Tokenization Details
    # 连续输入序列 (B, T, _) 到 [-1, 1] 区间
    if obs_continuous:
        c = mu_law_encode(x, mu, m)
        #  使用 1024 bins 离散化 并 shift 结果整数（到[32000,33024]）
        c = (c + 1) * (bins / 2)
        c = c.type(torch.int32)
        c += shift
    else:
        c = x + shift
    return c


def tokenize_act_values(x, bins=1024, act_continuous=True, shift=33024):
    # Appendix B. Agent Data Tokenization Details
    # 动作向量  (B, T, _) 已经位于 [-1, 1] 区间内
    if act_continuous:
        c = x
        #  使用 1024 bins 离散化 并 shift 结果整数（到[32000,33024]）
        c = (c + 1) * (bins / 2)
        c = c.type(torch.int32)
        c += shift
    else:
        c = x + shift
    return c


def inverse_tokenize_act_values(x, bins=1024, act_continuous=True, shift=33024):
    # 从 tokenize 的动作返回原动作
    if act_continuous:
        c = x - shift
        #  使用 1024 bins 离散化 并 shift 结果整数（到[32000,33024]）
        c = (2 * c) / bins - 1
        c = c.type(torch.float32)
    else:
        c = x - shift
    return c



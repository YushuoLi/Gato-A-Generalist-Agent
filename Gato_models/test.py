# import tensorflow as tf

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GatoConfig
# from tensorflow.keras import layers, models
from typing import Union, Dict, Any
from torch.distributions.uniform import Uniform
from Gato_models.embedding import _randomized_positions,_rounded_mean_positions

# -----------------------------------------------------------------------------------------------------------
# x = torch.tensor([1,2,3])
# print(x*6+1.0)
# y = torch.tensor([3,-2,1])
# print(x*y)


# -----------------------------------------------------------------------------------------------------------
# v = torch.tensor([1.0,2.0,1.0])
# j = torch.tensor([4.0,5.0,4.0])
# min_v = torch.zeros(v.shape)
# print(min_v)
# uni_dis = Uniform(min_v,j)
# print(uni_dis.sample())


# -----------------------------------------------------------------------------------------------------------
# a= torch.round(torch.unsqueeze(torch.tensor([-0.5, 0.5, 1.5, 2.5]), 1))
# print(a)
# a= torch.round(torch.tensor([-0.5, 0.5, 1.5, 2.5]))
# b = torch.ones((5,2))
# print(torch.reshape(b,(-1,)))


# -----------------------------------------------------------------------------------------------------------
# a,b=(
#     (1,3) if True else (2,3)
# )
# c= (
#     (1,3) if True else (2,3)
# )
# print(a,b,c)


# -----------------------------------------------------------------------------------------------------------
# net = nn.Embedding(128, 45)
# a = torch.arange(20)
# print(a, a.shape)
# print(net(a), net(a).shape)
# b = torch.ones((5,3,20,45))
# c = net(a)+b
# print(c, c.shape)

# -----------------------------------------------------------------------------------------------------------
# a = torch.tensor([1,2,3])
# b = torch.ones(3)
# print(torch.cat((a,b)))
# print(torch.arange(0,5))

# -----------------------------------------------------------------------------------------------------------
# max_T = 10
# ones = torch.ones((max_T, max_T))
# mask = torch.tril(ones).view(1, 1, 1, max_T, max_T)  # 生成对角线矩阵用作掩码
# # print(mask)
# # print(mask[..., :5, :5] == 0)
# print(mask[...,:5,:5] == mask[:,:,:, :5,:5])

# -----------------------------------------------------------------------------------------------------------
# a = torch.ones((3,2,4,7), dtype=torch.int32)
# d = torch.ones((3,2,7,4), dtype=torch.int32)
# b = torch.arange(168, dtype=torch.int32).reshape((3,2,7,4))
# c = a @ d
# print(c.shape)
# print(c[0][1],c[0][0])

# -----------------------------------------------------------------------------------------------------------
# sdf = torch.rand(1)
# idx = torch.randint(5, (8,2))
# print(idx)
# g = idx.view(16,-1)
# print(idx.shape)
# print(g.shape)

# -----------------------------------------------------------------------------------------------------------
# a = torch.ones((5,6,7))
# b = torch.ones((5,6,7))
# c = torch.ones((5,6,7))
# d = torch.stack((a,b,c), dim=1)
# g = torch.cat((a,b,c), dim=1)
# print(d.shape)
# print(g.shape)

# -----------------------------------------------------------------------------------------------------------
# B, T, H = 5, 6, 7
# a = torch.rand((B, T, H))  # (B, T, H)
# b = torch.rand((B, T, H))
# c = torch.rand((B, T, H))
# h = torch.stack((a, b, c), dim=1).permute(0, 2, 1, 3).reshape(B, 3 * T, H)
# h_2 = torch.cat((a, b, c), dim=-1).reshape(B, 3 * T, H)
# print(h==h_2)


# -----------------------------------------------------------------------------------------------------------
# a = torch.rand((B, T, H)).reshape((B,H, T))
# b = torch.rand((B, T, H)).reshape(B, H, T)
# print(a.shape, b.shape)

# -----------------------------------------------------------------------------------------------------------
# conv = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1)
# pif = torch.ones((5,4,3,64,64))
# x = conv(pif)
# print(x.shape)

# -----------------------------------------------------------------------------------------------------------
# a = torch.tensor([  0,   9,  18,  27,  37,  46,  55,  64,  73,  82,  91, 101, 110, 119], dtype=torch.int32)
# b = torch.tensor([  9,  18,  27,  37,  46,  55,  64,  73,  82,  91, 101, 110, 119, 128], dtype=torch.int32)
# # c = _randomized_positions(a, b)
# c = _rounded_mean_positions(a,b )
# print(c)
# print(c.type(torch.int32))

# -----------------------------------------------------------------------------------------------------------
# t = torch.ones((5,4))
# q = torch.ones((4,5))
# # print(t.view(-1,).shape)
# print(t @ q)
# print(torch.ones((5,4)).is_cuda)

# -----------------------------------------------------------------------------------------------------------
# x = torch.tensor([1,2,3,4])
# mu=100
# m=256
# sign = torch.sign(x)
# numerator = torch.log(torch.abs(x) * mu + 1.0)
# denominator = math.log(m * mu + 1.0)
# d = (numerator / denominator) * sign
# print(d)

# -----------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------



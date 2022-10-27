import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Gato_models.embedding import PatchPositionEncoding, ResidualEmbedding, LocalPositionEncoding, DiscreteEmbedding
from typing import Dict, Any, Union
from Gato_models.config import GatoConfig
from Gato_models.tokenizers import tokenize_act_values, tokenize_obs_values

class MaskedCausalAttention(nn.Module):
    """
    掩码自注意力机制
    """
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.W_Q = nn.Linear(h_dim, h_dim)  # 后一项可以更改？
        self.W_K = nn.Linear(h_dim, h_dim)
        self.W_V = nn.Linear(h_dim, h_dim)
        self.W_O = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.W_O_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)  # 生成对角线矩阵用作掩码

        # register buffer makes sure mask does not get updated during backpropagation
        self.register_buffer('mask', mask)



    def forward(self, x):
        B, T, H = x.shape # batch size, seq length, Q,K,V dim(=hidden dim)
        N, D = self.n_heads, H // self.n_heads  # N = num heads, D = attention dim, H = N * D

        # self.W_Q(x):(B,N,H), W_Q(x).view:(B, T, N, D), transpose: (B, N, T, D)
        q = self.W_Q(x).view(B, T, N, D).transpose(1, 2)
        k = self.W_K(x).view(B, T, N, D).transpose(1, 2)
        v = self.W_V(x).view(B, T, N, D).transpose(1, 2)

        # weights (B, N, T, T)
        attention_weights = q @ k.transpose(2, 3) / math.sqrt(D)
        # 对注意力权重进行掩码处理
        attention_weights = attention_weights.masked_fill(self.mask[..., :T, :T] == 0, float('-inf'))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(attention_weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N*D)
        out = self.W_O_drop(self.W_O(attention))

        return out


class Block(nn.Module):
    """
    Transformer 解码器 block
    """
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4 * h_dim),
                nn.GELU(),
                nn.Linear(4 * h_dim, h_dim),
                nn.Dropout(drop_p),
            )  # 逐点前馈神经网络
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x)  # residual
        x = self.ln1(x)
        x = x + self.mlp(x)  # residual
        x = self.ln2(x)
        return x


class DecisionTransformer(nn.Module):
    """
    处理 d4rl 数据的 DT 模型
    """
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096, state_syb='vec'):
        super().__init__()

        self.state_dim = state_dim  # 状态向量的维度
        self.act_dim = act_dim  # 动作向量的维度
        self.h_dim = h_dim  # 注意力机制之后的维度
        self.context_len = context_len  # 轨迹的最大长度（遍历过的timestep）
        self.max_timestep = max_timestep  # 轨迹数据记录下的最大时间步

        ### transformer blocks
        input_seq_len = 3 * context_len  # 输入序列的最大长度
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        # embedding层的输入:[batch_size,seq_len],输出:[batch_size,seq_len,embedding_size],得到每个序列（batch）每个单词的词向量
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)
        # # discrete actions
        # self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        # use_action_tanh = False # False for discrete actions

        # continuous actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        use_action_tanh = True  # True for continuous actions

        # prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        )


    def forward(self, timesteps, states, actions, returns_to_go):
        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings

        # stack rtg, states and actions and reshape sequence as (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        # [B, 3, T, H] -> [B, T, 3, H] -> [B, 3 * T, H]:(r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        h = torch.stack((returns_embeddings, state_embeddings, action_embeddings),
                        dim=1).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)

        h = self.embed_ln(h)

        # transformer and prediction
        h = self.transformer(h)

        # 修改 h 的形状, 使得 [B, 3 * T, H] -> [B, 3, T, H]
        # h[:, i] 的形状为 [B, T, H]
        # h[:, 0, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t, a_t
        # 即每个时间步 (t) 我们有 3 个 transformer 根据以前序列提取的特征输出
        # 以前序列包括 t 之前的所有时间步以及当前时刻 t 的 3 个输入 (r_t, s_t, a_t)
        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_rtg(h[:, 2])     # predict next rtg given r, s, a
        state_preds = self.predict_state(h[:, 0])    # predict next state given r, s, a
        action_preds = self.predict_action(h[:, 1])  # predict action given r, s

        return state_preds, action_preds, return_preds


class Gato(nn.Module):
    """
    处理控制问题的 Gato 模型
    """
    def __init__(self, state_dim, act_dim,
                 n_blocks, context_len, n_heads, drop_p,
                 config: Union[GatoConfig, Dict[str, Any]],
                 max_timestep=8096,
                 symbol='state_value'):
        super().__init__()
        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config

        self.device = self.config.device

        self.symbol = symbol

        self.state_dim = state_dim
        self.img_height = config.img_height  # 状态图片的维度 实际上模型不应该固定此项，就像在local embed中应适配各种情况
        self.img_width = config.img_width
        self.act_dim = act_dim
        self.embed_dim = self.config.layer_width
        self.h_dim = self.embed_dim  # 注意力机制之后的维度
        self.context_len = context_len  # 轨迹的最大长度（遍历过的最大timestep长度）
        self.max_timestep = max_timestep  # 轨迹数据记录下的最大时间步

        # embedding
        self.value_embed = DiscreteEmbedding(self.config)  # 状态、动作数值的 embedding table
        self.fig_embed = ResidualEmbedding(self.config)  # 图片 Resnet embedding
        self.patch_embed = PatchPositionEncoding(self.config)  # 图片位置 encoding
        self.local_embed = LocalPositionEncoding(self.config)  # 时间步内的局部位置 encoding
        # self.embed_timestep = nn.Embedding(max_timestep, self.h_dim)  # 时间步 embedding 一种针对时间步的 embedding

        # transformer blocks
        input_seq_len = self.config.token_sequence_length  # 输入序列的最大长度
        blocks = [Block(self.h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        # projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(self.h_dim)

        # prediction linear
        self.predict_action = nn.Sequential(nn.Linear(self.h_dim, 1),
                                            nn.Tanh(),
                                            )

    def sequence(self, timesteps, states, actions):
        symbol = self.symbol

        B, T = timesteps.shape  # (B, T)
        # timesteps: (B, T) -> (B, T, emd) -> (B, T, 1, emd)
        # print(torch.max(timesteps))
        # time_embeddings = self.embed_timestep(timesteps).unsqueeze(dim=2)  # 需要保证手工设置的maxtimestep足够大，不然会越界报错

        # 分隔符: (B, T) 输入 embedding 词典的最后一个
        separator = torch.ones((B, T), dtype=torch.int32).to(self.device) * (self.config.embedding_input_size-1)
        sep_emd = self.value_embed(separator).unsqueeze(dim=2)  # (B, T, emd) -> (B, T, 1, emd)

        emd = self.config.layer_width

        # 动作 actions: (B, T) or (B, T, A)
        a_shape = actions.shape
        if len(a_shape) == 2:  # (B, T)
            actions = torch.unsqueeze(actions, dim=-1)  # (B, T, 1)
            A = 1
        else:  # (B, T, A)
            # 动作序列的长度
            A = a_shape[2]
        # (B, T, A) -> (B, T * A)
        actions = actions.view(B, -1)
        # (B, T * A) -> (B, T * A, emd)
        action_embeddings = self.value_embed(actions)
        # (B, T * A, emd) -> (B, T, A, emd)
        action_emd = action_embeddings.reshape(B, T, -1, emd)

        # 状态
        if symbol == 'state_value':
            # states: (B, T, S) -> (B, T * S)
            states = states.view(B, -1)
            # (B, T * S) -> (B, T * S, emd)
            state_embeddings = self.value_embed(states)
            # (B, T * S, emd) -> (B, T, S, emd)
            state_emd = state_embeddings.reshape(B, T, -1, emd)
            # 状态序列的长度
            S = state_emd.shape[2]

        else:  # symbol == 'state_figure'
            # states: (B, T, C, H, W)
            B, T, C, H, W = states.shape
            # (B, T, C, H, W) -> (B * T, C, H, W)
            states = states.view(-1, C, H, W)
            # (B * T, C, H, W) -> (B * T, len, emd)
            state_patch_embeddings = self.fig_embed(states)
            # (B * T, len, emd) -> (B * T, len, emd)
            state_embeddings = self.patch_embed(state_patch_embeddings)
            # (B * T, len, emd) -> (B, T, len, emd)
            state_emd = state_embeddings.reshape(B, T, -1, emd)
            # 状态序列的长度
            S = state_emd.shape[2]

        # timestep embedding
        # state_emd += time_embeddings  # broadcast
        # action_emd += time_embeddings  # broadcast

        # 形成序列化轨迹 (B, T, S + 1 + A, emd) -> (B, T * (S + 1 + A), emd)
        h = torch.cat((state_emd, sep_emd, action_emd), dim=2)
        h = self.local_embed(h, S).reshape(B, -1, emd)
        return h, (B, T, S, A)

    def forward(self, timesteps, states, actions):
        symbol = self.symbol

        # ---------------------------------------------------- tokenize -----------------------------------------------
        if symbol == 'state_value':
            obs_continuous = states.dtype == torch.float32
            states = tokenize_obs_values(states,
                                         bins=self.config.observation_size,
                                         obs_continuous=obs_continuous,
                                         shift=self.config.vocabulary_size)
        act_continuous = actions.dtype == torch.float32
        actions = tokenize_act_values(actions,
                                      bins=self.config.actions_size,
                                      act_continuous=act_continuous,
                                      shift=self.config.vocabulary_size + self.config.observation_size)

        # ------------------------------------------- sequence + embedding ------------------------------------
        # 序列化 +  embedding
        h, (B, T, S, A) = self.sequence(timesteps, states, actions)
        # 层归一化
        h = self.embed_ln(h)
        # transformer and prediction
        h = self.transformer(h)

        # (B, T * (S + 1 + A), emd) -> (B, T, S + 1 + A, emd)
        h = h.reshape(B, T, -1, self.h_dim)

        # get predictions  (B, T, S + 1 + A, emd) -> (B, T, A, 1) -> (B, T, A)
        action_preds = self.predict_action(h[:, :, S+1:, :]).squeeze()

        return action_preds


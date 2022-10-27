import copy
from typing import Dict, Any


class GatoConfig:

    @staticmethod
    def large():
        return GatoConfig(num_transformer_blocks=24,
                          num_attention_heads=16,
                          layer_width=2048,
                          feedforward_hidden_size=8192,
                          key_value_size=128)

    @staticmethod
    def baseline():
        return GatoConfig(num_transformer_blocks=12,
                          num_attention_heads=12,
                          layer_width=1536,
                          feedforward_hidden_size=6144,
                          key_value_size=128)

    @staticmethod
    def small():
        return GatoConfig(num_transformer_blocks=8,
                          num_attention_heads=24,
                          layer_width=768,
                          feedforward_hidden_size=3072,
                          key_value_size=32)

    def __init__(self, **kwargs):
        self.device = kwargs.pop('device', 'cuda:0')

        # Section 2.3. Training
        # Max token sequence length 最大序列长度 T
        self.token_sequence_length = kwargs.pop('token_sequence_length', 1024)

        # Section 2.1. Tokenization
        # Text - SentencePiece 文本词汇表大小
        self.vocabulary_size = kwargs.pop('vocabulary_size', 32000)
        # Action vocabulary size 行动词汇表大小 (离散 + 连续） 离散值代表某个动作对应的值，连续值经过离散化也可以同理
        self.actions_size = kwargs.pop('actions_size', 1024)
        # Observation vocabulary size 观测词汇表大小
        self.observation_size = kwargs.pop('observation_size', 1024)

        # Appendix C.1. Transformer Hyperparameters
        self.num_transformer_blocks = kwargs.pop('num_transformer_blocks', 8)
        self.num_attention_heads = kwargs.pop('num_attention_heads', 24)
        self.layer_width = kwargs.pop('layer_width', 768)  # embedding_dim
        self.feedforward_hidden_size = kwargs.pop('feedforward_hidden_size', 3072)
        self.key_value_size = kwargs.pop('key_value_size', 32)

        # Appendix E. Regularization
        self.dropout_rate = kwargs.pop('dropout_rate', 0.1)

        # Appendix C.2. Embedding Function
        self.num_group_norm_groups = kwargs.pop('num_group_norm_groups', 32)
        self.img_patch_size = kwargs.pop('img_patch_size', 16)
        self.in_channels = kwargs.pop('in_channels', 4)
        self.out_channels = kwargs.pop('out_channels', 16)
        self.stride = kwargs.pop('stride', 1)
        self.img_height = kwargs.pop('img_height', 80)
        self.img_width = kwargs.pop('img_width', 64)


        # Appendix C.3. Position Encodings > Patch Position Encodings
        self.discretize_depth = kwargs.pop('discretize_depth', 128)
        # Appendix C.3. Position Encodings > Local Observation Position Encodings
        self.local_position_encoding_size = kwargs.pop('local_position_encoding_size', 512)  # 观测向量的个数不能超过

    @property
    def embedding_input_size(self):
        # 总 embedding table 的输入词表大小 非图像数据
        return self.vocabulary_size + self.observation_size + self.actions_size + 1

    @property
    def output_target_size(self):
        # 总 action_size 的输入词表大小
        return self.vocabulary_size + self.actions_size

    def to_dict(self) -> Dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        return output

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GatoConfig":
        config = cls(**config_dict)
        return config
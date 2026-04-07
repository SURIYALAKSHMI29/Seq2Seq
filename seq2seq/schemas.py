from dataclasses import dataclass
from typing import Optional
from hydra.core.config_store import ConfigStore


@dataclass
class BaseEncoderConfig:
    model_name: str
    vocab_size: int
    embed_size: int
    layers: int = 1
    embed_dropout: float = 0.0
    category: str = "encoder"
    max_src_len: int = 30


@dataclass(kw_only=True)
class LSTMEncoderConfig(BaseEncoderConfig):
    hidden_size: int
    model_name: str = "lstm"
    bidirectional: Optional[bool] = False


@dataclass(kw_only=True)
class TransformerEncoderConfig(BaseEncoderConfig):
    model_name: str = "transformer"
    ffn_multiplier: int = 4
    num_heads: int = 1
    attn_dropout: float = 0.0
    ffn_dropout: float = 0.0


@dataclass
class BaseDecoderConfig:
    model_name: str
    vocab_size: int
    embed_size: int
    layers: int = 1
    embed_dropout: float = 0.0
    fc_dropout: float = 0.0
    category: str = "decoder"
    max_trg_len: int = 30


@dataclass(kw_only=True)
class LSTMDecoderConfig(BaseDecoderConfig):
    hidden_size: int
    model_name: str = "lstm"
    bidirectional: Optional[bool] = True
    attention: bool = True
    teacher_forcing_ratio: float = 0.5


@dataclass(kw_only=True)
class TransformerDecoderConfig(BaseDecoderConfig):
    model_name: str = "transformer"
    num_heads: int = 1
    ffn_multiplier: int = 4
    self_attn_dropout: float = 0.0
    cross_attn_dropout: float = 0.0
    ffn_dropout: float = 0.0


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 0.001
    epochs: int = 1


@dataclass
class PathsConfig:
    train: str
    val: str
    src_vocab: str
    trg_vocab: str
    model: str
    raw_data: str


@dataclass
class DataConfig:
    PAD: int = 0
    MAX_LEN: int = 10
    NUM_PREFIXES: int = 10
    src_vocab_size: int = 5000
    trg_vocab_size: int = 5000


@dataclass
class Config:
    encoder: BaseEncoderConfig
    decoder: BaseDecoderConfig
    train: TrainConfig
    paths: PathsConfig
    data: DataConfig


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

cs.store(group="encoder", name="lstm", node=LSTMEncoderConfig)
cs.store(group="encoder", name="transformer", node=TransformerEncoderConfig)

cs.store(group="decoder", name="lstm", node=LSTMDecoderConfig)
cs.store(group="decoder", name="transformer", node=TransformerDecoderConfig)

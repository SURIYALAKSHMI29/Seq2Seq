from dataclasses import dataclass
from typing import Optional
from hydra.core.config_store import ConfigStore


@dataclass
class EncoderConfig:
    model_name: str
    hidden_size: int
    vocab_size: int
    embed_size: int
    layers: Optional[int] = 1
    bidirectional: Optional[bool] = False
    category: str = "encoder"
    max_src_len: Optional[int] = 10


@dataclass
class DecoderConfig:
    model_name: str
    hidden_size: int
    vocab_size: int
    embed_size: int
    layers: Optional[int] = 1
    bidirectional: Optional[bool] = False
    attention: Optional[bool] = True
    teacher_forcing: Optional[bool] = True
    teacher_forcing_ratio: Optional[float] = 0.5
    category: str = "decoder"
    max_trg_len: Optional[int] = 12


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


@dataclass
class Config:
    encoder: EncoderConfig
    decoder: DecoderConfig
    train: TrainConfig
    paths: PathsConfig


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

from dataclasses import dataclass, field
from typing import Optional, Tuple

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class EmbeddingConfig:
    dim: int = MISSING


@dataclass
class RegularizationConfig:
    embed_dropout: float = 0.0


## Encoder sub-configs
@dataclass
class LSTMArchConfig:
    hidden_size: int = MISSING
    layers: int = 1
    bidirectional: bool = False


@dataclass
class TransformerAttentionConfig:
    layers: int = MISSING
    num_heads: int = MISSING
    ffn_multiplier: int = 4
    activation: str = "relu"


@dataclass
class LSTMEncoderRegConfig(RegularizationConfig):
    pass  # only embed_dropout for encoder LSTM


@dataclass
class TransformerEncoderRegConfig(RegularizationConfig):
    attn_dropout: float = 0.0
    ffn_dropout: float = 0.0


## Decoder sub-configs
@dataclass
class LSTMDecoderArchConfig(LSTMArchConfig):
    bidirectional: bool = True  # decoder default differs from encoder
    attention: bool = True
    teacher_forcing_ratio: float = 0.5


@dataclass
class LSTMDecoderRegConfig(RegularizationConfig):
    fc_dropout: float = 0.0


@dataclass
class TransformerDecoderRegConfig(RegularizationConfig):
    self_attn_dropout: float = 0.0
    cross_attn_dropout: float = 0.0
    ffn_dropout: float = 0.0
    fc_dropout: float = 0.0


@dataclass
class BaseCmpConfig:
    """Shared identity fields for any encoder or decoder."""

    model_name: str = MISSING
    vocab_size: int = MISSING
    max_len: int = 30
    category: str = MISSING


## encoder
@dataclass
class LSTMEncoderConfig(BaseCmpConfig):
    model_name: str = "lstm"
    category: str = "encoder"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    architecture: LSTMArchConfig = field(default_factory=LSTMArchConfig)
    regularization: LSTMEncoderRegConfig = field(default_factory=LSTMEncoderRegConfig)


@dataclass
class TransformerEncoderConfig(BaseCmpConfig):
    model_name: str = "transformer"
    category: str = "encoder"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    attention: TransformerAttentionConfig = field(
        default_factory=TransformerAttentionConfig
    )
    regularization: TransformerEncoderRegConfig = field(
        default_factory=TransformerEncoderRegConfig
    )


## decoder
@dataclass
class LSTMDecoderConfig(BaseCmpConfig):
    model_name: str = "lstm"
    category: str = "decoder"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    architecture: LSTMDecoderArchConfig = field(default_factory=LSTMDecoderArchConfig)
    regularization: LSTMDecoderRegConfig = field(default_factory=LSTMDecoderRegConfig)


@dataclass
class TransformerDecoderConfig(BaseCmpConfig):
    model_name: str = "transformer"
    category: str = "decoder"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    attention: TransformerAttentionConfig = field(
        default_factory=TransformerAttentionConfig
    )
    regularization: TransformerDecoderRegConfig = field(
        default_factory=TransformerDecoderRegConfig
    )


## Optimizer
@dataclass
class OptimizerKwargsConfig:
    lr: float = 2.0e-4
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.98)
    eps: float = 1.0e-9


@dataclass
class OptimizerConfig:
    name: str = "Adam"
    kwargs: OptimizerKwargsConfig = field(default_factory=OptimizerKwargsConfig)


## Trainer
@dataclass
class TrainerConfig:
    max_epochs: int = 10


## data
@dataclass
class VocabConfig:
    MAX_LEN: int = 10
    NUM_PREFIXES: int = 10
    PAD: int = 0
    src_vocab_size: int = 5000
    trg_vocab_size: int = 5000
    src_vocab: str = MISSING
    trg_vocab: str = MISSING


@dataclass
class SplitDataConfig:
    data_path: str = MISSING


@dataclass
class DataConfig:
    batch_size: int = 32
    raw_data_path: str = MISSING
    vocab: VocabConfig = field(default_factory=VocabConfig)
    train: SplitDataConfig = field(default_factory=SplitDataConfig)
    val: SplitDataConfig = field(default_factory=SplitDataConfig)


## model
@dataclass
class ModelConfig:
    encoder: str = MISSING
    decoder: str = MISSING
    saved_model_path: str = MISSING


## Root config
@dataclass
class Config:
    encoder: BaseCmpConfig = MISSING
    decoder: BaseCmpConfig = MISSING
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


## BERT Config
@dataclass
class BERTDataConfig:
    vocab_size: int = 15
    max_src_len: int = 10
    num_segments: int = 2
    batch: int = 32


@dataclass
class BERTModelConfig:
    saved_model_path: str = MISSING


@dataclass
class BERTRegularizationConfig(RegularizationConfig):
    attn_dropout: float = 0.0
    ffn_dropout: float = 0.0


# BERT Root
@dataclass
class BERTConfig:
    data: BERTDataConfig = field(default_factory=BERTDataConfig)
    model: BERTModelConfig = field(default_factory=BERTModelConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    attention: TransformerAttentionConfig = field(
        default_factory=TransformerAttentionConfig
    )
    regularization: BERTRegularizationConfig = field(
        default_factory=BERTRegularizationConfig
    )
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


## ConfigStore registration

cs = ConfigStore.instance()

cs.store(name="base_config", node=Config)

cs.store(group="encoder", name="lstm", node=LSTMEncoderConfig)
cs.store(group="encoder", name="transformer", node=TransformerEncoderConfig)

cs.store(group="decoder", name="lstm", node=LSTMDecoderConfig)
cs.store(group="decoder", name="transformer", node=TransformerDecoderConfig)

cs.store(name="bert_config", node=BERTConfig)

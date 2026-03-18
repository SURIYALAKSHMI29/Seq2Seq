from dataclasses import dataclass
from typing import Optional


@dataclass
class EncoderConfig:
    category: str
    model_name: str
    hidden_size: int
    vocab_size: int
    embed_size: int
    layers: Optional[int] = 1
    bidirectional: Optional[bool] = False


@dataclass
class DecoderConfig:
    category: str
    model_name: str
    hidden_size: int
    vocab_size: int
    embed_size: int
    layers: Optional[int] = 1
    bidirectional: Optional[bool] = False
    attention: Optional[bool] = True

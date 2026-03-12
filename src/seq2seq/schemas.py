from dataclasses import dataclass
from typing import Optional


@dataclass
class EncoderConfig:
    type: str
    hidden_size: int
    vocab_size: int
    embed_size: int
    layers: Optional[int] = 1
    bidirectional: Optional[bool] = False


@dataclass
class DecoderConfig:
    type: str
    hidden_size: int
    vocab_size: int
    embed_size: int
    layers: Optional[int] = 1
    bidirectional: Optional[bool] = False
    attention: Optional[bool] = True

from typing import Type

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from seq2seq.modules.attention import MultiHeadAttention
from seq2seq.modules.embedding import positional_encoding
from seq2seq.registry import REGISTRY
from seq2seq.schemas import BaseCmpConfig


class Encoder(nn.Module):

    def __init__(self, config: BaseCmpConfig):
        super().__init__()
        self.config = config
        self.embedding: nn.Embedding = nn.Embedding(
            config.vocab_size, config.embed_size
        )
        self.enc_module: nn.Module | None = None

        self.embed_dropout: nn.Dropout | None = None


@REGISTRY.register("encoder", "lstm")
class LSTMEncoder(Encoder):
    def __init__(self, config: BaseCmpConfig):
        super().__init__(config)
        self.enc_module = nn.LSTM(
            config.embed_size,
            config.hidden_size,
            config.layers,
            batch_first=True,
            bidirectional=config.bidirectional,
        )

        self.embed_dropout = nn.Dropout(config.embed_dropout)

    @property
    def output_dim(self):
        return self.config.hidden_size * (2 if self.config.bidirectional else 1)

    def forward(self, x: Tensor, lengths: Tensor):
        # print(x.shape)  # batch, seq_len
        x: Tensor = self.embed_dropout(self.embedding(x))  # batch, seq_len, embed

        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_outputs, (hidden, cell) = self.enc_module(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs, batch_first=True
        )  # outputs = (batch, src_len, hidden)

        return {"outputs": outputs, "hidden": hidden, "cell": cell}


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_dropout: float,
        ffn_dropout: float,
        ffn_multiplier: int,
        activation: Type[nn.Module],
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_multiplier * d_model),
            activation(),
            nn.Linear(ffn_multiplier * d_model, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.ffn_dropout = nn.Dropout(ffn_dropout)

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
    ):
        # normalized_x = self.norm1(x)
        # attn_out_norm = self.self_attn(
        #     query=normalized_x,
        #     key=normalized_x,
        #     value=normalized_x,
        #     mask=mask,
        # )
        # x = x + self.attn_dropout(attn_out_norm)

        attn_out = self.self_attn(
            query=x,
            key=x,
            value=x,
            mask=mask,
        )
        x = self.norm1(x + self.attn_dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.ffn_dropout(ffn_out))

        return x


@REGISTRY.register("encoder", "transformer")
class TransformerEncoder(Encoder):
    def __init__(self, config: BaseCmpConfig):
        super().__init__(config)
        self.activation = self.get_activation(config.attention.activation.lower())
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    config.embed_size,
                    config.num_heads,
                    config.attn_dropout,
                    config.ffn_dropout,
                    config.ffn_multiplier,
                    self.activation,
                )
                for _ in range(config.layers)
            ]
        )

        self.embed_dropout = nn.Dropout(config.embed_dropout)

        self.register_buffer(
            "pe", positional_encoding(config.max_src_len, config.embed_size)
        )

    @property
    def output_dim(self):
        return self.config.embed_size

    def get_activation(self, activation_name):
        ACTIVATIONS = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }

        if activation_name not in ACTIVATIONS:
            raise ValueError(
                f"Activation {activation_name} not in {list(ACTIVATIONS.keys())}"
            )

        return ACTIVATIONS[activation_name]

    def forward(self, x: Tensor, lengths: Tensor):
        # print("raw x", x.shape)
        batch, max_src_len = x.shape

        # print(x.shape)  # batch, seq_len
        x = self.embed_dropout(self.embedding(x))  # batch, seq_len, embed
        # print("x embeddings", x.shape)

        ## scaled up -> embeddings dominates
        x = x * (self.config.embed_size**0.5)

        x = x + self.pe[:, :max_src_len, :]

        mask = (
            (
                torch.arange(max_src_len).expand(batch, max_src_len)
                < lengths.unsqueeze(1)
            )
            .unsqueeze(1)
            .unsqueeze(1)
        ).to(x.device)
        # batch, 1, 1, max_src_len

        # print("mask", mask.shape)

        for layer in self.layers:
            x = layer(x, mask)

        return {"outputs": x}

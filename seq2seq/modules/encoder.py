import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from seq2seq.registry import REGISTRY
from seq2seq.schemas import EncoderConfig
from seq2seq.modules.attention import Attention
from seq2seq.modules.encoding import positional_encoding


class Encoder(nn.Module):

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.embedding: nn.Embedding = nn.Embedding(
            config.vocab_size, config.embed_size
        )
        self.enc_module: nn.Module | None = None

        self.dropout = nn.Dropout(0.2)


@REGISTRY.register("encoder", "lstm")
class LSTMEncoder(Encoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        self.enc_module = nn.LSTM(
            config.embed_size,
            config.hidden_size,
            config.layers,
            batch_first=True,
            bidirectional=config.bidirectional,
        )

    @property
    def output_dim(self):
        return self.config.hidden_size * (2 if self.config.bidirectional else 1)

    def forward(self, x: Tensor, lengths: Tensor):
        # print(x.shape)  # batch, seq_len
        x: Tensor = self.dropout(self.embedding(x))  # batch, seq_len, embed

        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_outputs, (hidden, cell) = self.enc_module(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs, batch_first=True
        )  # outputs = (batch, src_len, hidden)

        return {"outputs": outputs, "hidden": hidden, "cell": cell}


@REGISTRY.register("encoder", "attention")
class AttentionEncoder(Encoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        self.enc_module = Attention(config.embed_size)
        self.pe = positional_encoding(config.max_src_len, config.embed_size)

        self.ffn = nn.Sequential(
            nn.Linear(config.embed_size, 4 * config.embed_size),
            nn.ReLU(),
            nn.Linear(4 * config.embed_size, config.embed_size),
        )

        self.norm1 = nn.LayerNorm(config.embed_size)
        self.norm2 = nn.LayerNorm(config.embed_size)

    @property
    def output_dim(self):
        return self.config.embed_size

    def forward(self, x: Tensor, lengths: Tensor):
        # print("raw x", x.shape)
        batch, max_src_len = x.shape

        # print(x.shape)  # batch, seq_len
        x = self.dropout(self.embedding(x))  # batch, seq_len, embed
        # print("x embeddings", x.shape)

        x = x + self.pe[:, :max_src_len, :]

        mask = (
            torch.arange(max_src_len).expand(batch, max_src_len) < lengths.unsqueeze(1)
        ).unsqueeze(1)
        # batch, 1, max_src_len

        # print("mask", mask.shape)

        attn_out = self.enc_module(
            query=x,
            key=x,
            value=x,
            mask=mask,
        )

        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # print("context:", context.shape)  # batch, seq_len, hidden_size
        # print("context:", x[0, :5, :])
        return {"outputs": x}

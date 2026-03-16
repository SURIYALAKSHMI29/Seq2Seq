import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from seq2seq.registry import ENCODERS
from seq2seq.schemas import EncoderConfig


class Encoder(nn.Module):

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.embedding: nn.Embedding = nn.Embedding(
            config.vocab_size, config.embed_size
        )
        self.enc_module: nn.Module | None = None


@ENCODERS.register("lstm")
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

    def forward(self, x: Tensor, lengths: Tensor):
        # print(x.shape)  # batch, seq_len
        x: Tensor = self.embedding(x)  # batch, seq_len, embed

        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_outputs, (hidden, cell) = self.enc_module(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs, batch_first=True
        )  # outputs = (batch, src_len, hidden)

        return outputs, (hidden, cell)

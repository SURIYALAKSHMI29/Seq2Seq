import torch
import torch.nn as nn
from torch import Tensor

from seq2seq.modules.attention import dot_attention
from seq2seq.registry import DECODERS
from seq2seq.schemas import DecoderConfig


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embed_size)
        self.dec_module: nn.Module | None = None
        self.attention = dot_attention
        linear_in = config.hidden_size
        if config.attention:
            linear_in *= 2
        if config.bidirectional:
            linear_in *= 2
        self.fc = nn.Linear(linear_in, config.vocab_size)


@DECODERS.register("lstm")
class LSTMDecoder(Decoder):

    def __init__(self, config: DecoderConfig):
        super().__init__(config)
        self.dec_module = nn.LSTM(
            config.embed_size,
            config.hidden_size,
            batch_first=True,
            bidirectional=config.bidirectional,
        )

    def forward(
        self,
        trg_input: Tensor,
        hidden_cell: tuple,
        encoder_hiddens: Tensor,
        src_lengths: Tensor,
    ):
        _, trg_len = trg_input.shape

        outputs = []

        for t in range(0, trg_len):
            # print("processing timestep", t)
            input = trg_input[:, 0].unsqueeze(1)

            input_embed = self.embedding(input)  # batch, 1, embed

            output, hidden_cell = self.dec_module(input_embed, hidden_cell)
            # output = (batch, 1, hidden)
            # hidden = (num_layers, batch, hidden)

            output = output.squeeze(1)  # batch, hidden
            if self.config.attention:
                hidden, _ = hidden_cell
                context, weights = self.attention(hidden, encoder_hiddens, src_lengths)
                # context = (batch, hidden)
                # weights = (batch, src_len)

                output = torch.cat((output, context), dim=1)

            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        print(outputs.shape)
        logits = self.fc(outputs)  # batch, trg_len, vocab

        # print(outputs)
        # print(type(logits), logits.shape)
        # print("Target length", trg_len)

        return logits

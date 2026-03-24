import torch
import torch.nn as nn
from torch import Tensor

from seq2seq.modules.attention import dot_attention
from seq2seq.registry import REGISTRY
from seq2seq.schemas import DecoderConfig


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embed_size)
        self.dec_module: nn.Module | None = None
        self.attention = dot_attention
        linear_in = config.hidden_size
        if config.bidirectional:
            linear_in *= 2
        if config.attention:
            linear_in += config.hidden_size
        self.fc = nn.Linear(linear_in, config.vocab_size)

        # self.fc_proj = nn.Sequential(
        #     nn.Linear(linear_in, config.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_size, config.embed_size),
        # )

        self.embed_dropout = nn.Dropout(0.1)
        self.fc_dropout = nn.Dropout(0.2)


@REGISTRY.register("decoder", "lstm")
class LSTMDecoder(Decoder):

    def __init__(self, config: DecoderConfig):
        super().__init__(config)
        self.dec_module = nn.LSTM(
            config.embed_size,
            config.hidden_size,
            config.layers,
            batch_first=True,
            bidirectional=config.bidirectional,
        )

    def forward(
        self,
        trg_input: Tensor,
        encoder_hiddens: Tensor,
        src_lengths: Tensor,
        teacher_forcing: bool,
        hidden_cell: tuple | None,
    ):
        _, trg_len = trg_input.shape

        outputs = []

        if hidden_cell is None:
            h_0 = (
                encoder_hiddens.mean(dim=1)
                .unsqueeze(0)
                .repeat(self.config.layers, 1, 1)
            )
            c_0 = torch.zeros_like(h_0)
        else:
            h_0, c_0 = hidden_cell
            if c_0 is None:
                c_0 = torch.zeros_like(h_0)

        hidden_cell = (h_0, c_0)

        batch, max_src_len, _ = encoder_hiddens.shape

        for t in range(trg_len):
            # print("processing timestep", t)
            if t == 0 or teacher_forcing:
                input = trg_input[:, t].unsqueeze(1)
            else:
                input = output.argmax(-1).unsqueeze(1)

            input_embed = self.embed_dropout(self.embedding(input))  # batch, 1, embed

            output, hidden_cell = self.dec_module(input_embed, hidden_cell)
            # output = (batch, 1, hidden)
            # hidden = (num_layers, batch, hidden)

            output = output.squeeze(1)  # batch, hidden
            if self.config.attention:
                hidden, _ = hidden_cell
                hidden = hidden[-1].unsqueeze(1)

                mask = (
                    torch.arange(max_src_len).expand(batch, max_src_len)
                    < src_lengths.unsqueeze(1)
                ).unsqueeze(1)

                # using dot_attention, key must be transposed
                context = self.attention(
                    query=hidden,
                    key=encoder_hiddens.transpose(1, 2),
                    value=encoder_hiddens,
                    mask=mask,
                ).squeeze(1)
                # context = (batch, hidden)

                output = torch.cat((output, context), dim=1)

            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        # fc_proj = self.fc_proj(self.fc_dropout(outputs))
        # logits = fc_proj @ self.embedding.weight.T  # batch, trg_len, vocab

        logits = self.fc_dropout(self.fc(outputs))  # batch, trg_len, vocab

        # print(outputs)
        # print(type(logits), logits.shape)
        # print("Target length", trg_len)

        # print("Logits shape", logits.shape)
        return logits

import torch
import torch.nn as nn
from torch import Tensor
import random

from seq2seq.modules.attention import dot_attention
from seq2seq.registry import REGISTRY
from seq2seq.schemas import DecoderConfig
from seq2seq.modules.attention import Attention
from seq2seq.modules.encoding import positional_encoding


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config

        self.tf_ratio = config.teacher_forcing_ratio

        self.embedding = nn.Embedding(config.vocab_size, config.embed_size)
        self.dec_module: nn.Module | None = None
        self.fc: nn.Module | None = None

        # self.fc_proj = nn.Sequential(
        #     nn.Linear(linear_in, config.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_size, config.embed_size),
        # )

        self.embed_dropout = nn.Dropout(0.1)
        self.fc_dropout = nn.Dropout(0.2)

    @property
    def needs_hidden(self):
        return False


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

        self.attention = dot_attention
        linear_in = config.hidden_size
        if config.bidirectional:
            linear_in *= 2
        if config.attention:
            linear_in += config.hidden_size
        self.fc = nn.Linear(linear_in, config.vocab_size)

    @property
    def input_dim(self):
        return self.config.hidden_size

    @property
    def needs_hidden(self):
        return True

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

        h_0, c_0 = hidden_cell
        if h_0 is None:
            h_0 = (
                encoder_hiddens.mean(dim=1)
                .unsqueeze(0)
                .repeat(self.config.layers, 1, 1)
            )
        if c_0 is None:
            c_0 = torch.zeros_like(h_0)

        hidden_cell = (h_0, c_0)

        batch, max_src_len, _ = encoder_hiddens.shape

        for t in range(trg_len):
            # print("processing timestep", t)
            use_teacher_forcing = teacher_forcing and (random.random() < self.tf_ratio)

            if t == 0 or use_teacher_forcing:
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


@REGISTRY.register("decoder", "attention")
class AttentionDecoder(Decoder):
    def __init__(self, config: DecoderConfig):
        super().__init__(config)
        self.dec_module = Attention(config.embed_size)
        self.cross_attn = Attention(config.embed_size)

        self.pe = positional_encoding(config.max_trg_len, config.embed_size)

        self.ffn = nn.Sequential(
            nn.Linear(config.embed_size, 4 * config.embed_size),
            nn.ReLU(),
            nn.Linear(4 * config.embed_size, config.embed_size),
        )

        self.fc = nn.Linear(config.embed_size, config.vocab_size)

        self.norm1 = nn.LayerNorm(config.embed_size)
        self.norm2 = nn.LayerNorm(config.embed_size)
        self.norm3 = nn.LayerNorm(config.embed_size)

    @property
    def input_dim(self):
        return self.config.embed_size

    def get_look_ahead_mask(self, trg_len):
        mask = torch.ones(trg_len, trg_len)
        mask = torch.tril(mask).bool()
        return mask.unsqueeze(0)  # (1, trg_len, trg_len)

    def forward(
        self,
        trg_input: Tensor,
        encoder_hiddens: Tensor,
        src_lengths: Tensor,
        *args,
        **kwargs
    ):
        # print("raw x", x.shape)
        batch, trg_len = trg_input.shape
        max_src_len = encoder_hiddens.shape[1]

        # print(x.shape)  # batch, seq_len
        x = self.embed_dropout(self.embedding(trg_input))  # batch, seq_len, embed
        # print("x embeddings", x.shape)

        ## positional encoding
        x = x + self.pe[:, :trg_len, :]

        look_ahead_mask = self.get_look_ahead_mask(trg_len)  # 1, trg_len, trg_len
        trg_pad_mask = (trg_input != 0).unsqueeze(1)  # batch, 1, trg_len
        trg_mask = trg_pad_mask & look_ahead_mask
        # print("trg_mask", trg_mask.shape)   # batch, trg_len, trg_len

        ## self - attention
        self_attn_out = self.dec_module(
            query=x,
            key=x,
            value=x,
            mask=trg_mask,
        )

        ## residual connection
        x = self.norm1(x + self_attn_out)

        ## cross attention
        src_mask = (
            torch.arange(max_src_len).expand(batch, max_src_len)
            < src_lengths.unsqueeze(1)
        ).unsqueeze(1)

        cross_attn_out = self.cross_attn(
            query=x,
            key=encoder_hiddens,
            value=encoder_hiddens,
            mask=src_mask,
        )

        x = self.norm2(x + cross_attn_out)

        ## Feed forward network
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)

        # print("context:", context.shape)  # batch, seq_len, hidden_size
        # print("context:", x[0, :5, :])

        logits = self.fc_dropout(self.fc(x))
        return logits

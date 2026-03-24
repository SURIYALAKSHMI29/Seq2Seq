import torch
import torch.nn as nn
from torch import Tensor

from seq2seq.schemas import EncoderConfig, DecoderConfig
from seq2seq.registry import REGISTRY


class Seq2Seq(nn.Module):

    def __init__(self, encoder_config: EncoderConfig, decoder_config: DecoderConfig):
        super().__init__()
        self.encoder = REGISTRY.build(encoder_config)
        self.decoder = REGISTRY.build(decoder_config)

        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

        enc_dim = self.encoder.output_dim
        dec_dim = self.decoder.input_dim

        self.enc_out_proj: nn.Module = None
        self.state_proj: nn.Module = None

        if enc_dim != dec_dim:
            self.enc_out_proj = nn.Linear(enc_dim, dec_dim)
            self.state_proj = nn.Linear(enc_dim, dec_dim)

        # if encoder_config.bidirectional:
        #     self.enc_out_proj = nn.Linear(
        #         encoder_config.hidden_size * 2, encoder_config.hidden_size
        #     )
        #     self.state_proj = nn.Linear(
        #         encoder_config.hidden_size * 2, encoder_config.hidden_size
        #     )

        # if encoder_config.hidden_size != decoder_config.hidden_size:
        #     self.enc_out_hidden_proj = nn.Linear(
        #         encoder_config.hidden_size, decoder_config.hidden_size
        #     )
        #     self.state_hidden_proj = nn.Linear(
        #         encoder_config.hidden_size, decoder_config.hidden_size
        #     )

    def transform_encoder_state(self, outputs, hidden, cell):
        ## transform encoder state if bidirectional
        # if self.encoder_config.bidirectional:

        #     # outputs are used in decoder attn
        #     # decoder hidden size = hidden size, to match the dimensions in dot prod, output is transformed
        #     outputs = self.enc_out_proj(outputs)

        #     # project 2H → H, only if decoder is not bidirectional
        #     if (
        #         hasattr(self.decoder_config, "bidirectional")
        #         and not self.decoder_config.bidirectional
        #     ):
        #         num_layers, batch, hidden_size = hidden.shape
        #         num_layers = num_layers // 2

        #         # reshape to separate directions (frwd & bwd)
        #         hidden = hidden.view(num_layers, 2, batch, hidden_size)

        #         # concatenate frwd + bwd
        #         hidden = torch.cat((hidden[:, 0], hidden[:, 1]), dim=2)

        #         # project 2H → H
        #         hidden = self.state_proj(hidden)

        #         if cell is not None:
        #             cell = cell.view(num_layers, 2, batch, hidden_size)
        #             cell = torch.cat((cell[:, 0], cell[:, 1]), dim=2)
        #             cell = self.state_proj(cell)

        # if hasattr(self, "enc_out_hidden_proj"):
        #     outputs = self.enc_out_hidden_proj(outputs)

        #     if hidden is not None:
        #         hidden = self.state_hidden_proj(hidden)

        #     if cell is not None:
        #         cell = self.state_hidden_proj(cell)

        if self.enc_out_proj is not None:
            outputs = self.enc_out_proj(outputs)

            if hidden is not None and self.encoder_config.bidirectional:
                num_layers, batch, hidden_size = hidden.shape
                num_layers = num_layers // 2

                # reshape to separate directions (frwd & bwd)
                hidden = hidden.view(num_layers, 2, batch, hidden_size)

                # concatenate frwd + bwd
                hidden = torch.cat((hidden[:, 0], hidden[:, 1]), dim=2)

                # project 2H → H
                hidden = self.state_proj(hidden)

                if cell is not None:
                    cell = cell.view(num_layers, 2, batch, hidden_size)
                    cell = torch.cat((cell[:, 0], cell[:, 1]), dim=2)
                    cell = self.state_proj(cell)

        return outputs, hidden, cell

    def adjust_layers(self, outputs, hidden, cell):
        ## if enc and dec have same num of layers -> no issue
        # if enc_layers > dec_layers -> return top dec_layers
        # if enc_layers < dec_layers -> repeat last layer

        enc_layers = hidden.size(0)
        dec_layers = self.decoder_config.layers

        if enc_layers > dec_layers:
            hidden = hidden[-dec_layers:]
            cell = cell[-dec_layers:]

        elif enc_layers < dec_layers:
            repeat = dec_layers - enc_layers
            hidden_extra = hidden[-1:].repeat(repeat, 1, 1)
            cell_extra = cell[-1:].repeat(repeat, 1, 1)

            hidden = torch.cat([hidden, hidden_extra], dim=0)
            cell = torch.cat([cell, cell_extra], dim=0)

        return outputs, hidden, cell

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_lengths: Tensor,
        teacher_forcing: bool = True,
    ):

        enc = self.encoder(src, src_lengths)

        encoder_outputs = enc["outputs"]
        hidden = enc.get("hidden")
        cell = enc.get("cell")

        # if bidirectional, transform encoder state
        encoder_outputs, hidden, cell = self.transform_encoder_state(
            encoder_outputs, hidden, cell
        )

        if hidden is not None and self.decoder.needs_hidden:
            # matching encoder and decoder layers
            encoder_outputs, hidden, cell = self.adjust_layers(
                encoder_outputs, hidden, cell
            )

        hidden_cell = (hidden, cell) if self.decoder.needs_hidden else None

        logits = self.decoder(
            trg, encoder_outputs, src_lengths, teacher_forcing, hidden_cell
        )
        return logits  # batch, trg_len, vocab

import torch
import torch.nn as nn
from torch import Tensor

from seq2seq.schemas import EncoderConfig, DecoderConfig
from seq2seq.registry import ENCODERS, DECODERS


class Seq2Seq(nn.Module):

    def __init__(self, encoder_config: EncoderConfig, decoder_config: DecoderConfig):
        super().__init__()
        self.encoder = ENCODERS.build(encoder_config)
        self.decoder = DECODERS.build(decoder_config)

    def forward(self, src: Tensor, trg: Tensor, src_lengths: Tensor):

        encoder_outputs, (hidden_cell) = self.encoder(src, src_lengths)
        # print("Hidden cell type", type(hidden_cell))

        outputs = self.decoder(trg, (hidden_cell), encoder_outputs, src_lengths)
        return outputs

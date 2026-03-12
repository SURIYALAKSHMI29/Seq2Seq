import torch
import pytest
import logging

from seq2seq.registry import DECODERS
from seq2seq.schemas import DecoderConfig
from seq2seq.modules.attention import dot_attention
from configs.seq2seq_config import DECODER_CONFIG

logger = logging.getLogger(__name__)


@pytest.fixture
def decoder_config():
    return DecoderConfig(
        type="lstm",
        vocab_size=10,
        embed_size=8,
        hidden_size=16,
        attention=False,
        bidirectional=False,
    )


@pytest.fixture
def decoder(decoder_config):
    torch.manual_seed(0)
    return DECODERS.build(decoder_config)


@pytest.fixture
def sample_trg(decoder_config):
    batch, seq_len = 4, 7
    trg_input = torch.randint(0, decoder_config.vocab_size, (batch, seq_len))
    src_lengths = torch.tensor([seq_len] * batch)

    num_directions = 2 if decoder_config.bidirectional else 1
    hidden = torch.zeros(1 * num_directions, batch, decoder_config.hidden_size)
    cell = torch.zeros_like(hidden)

    encoder_hiddens = torch.randn(batch, seq_len, decoder_config.hidden_size)

    return {
        "trg_input": trg_input,
        "hidden_cell": (hidden, cell),
        "encoder_hiddens": encoder_hiddens,
        "src_lengths": src_lengths,
    }


def test_decoder_make(decoder_config):
    decoder_config.type = "rnn"
    with pytest.raises(Exception) as e:
        DECODERS.build(decoder_config)
    logger.info("Raised exception: %s", e.value)


def test_decoder_forward(decoder, decoder_config, sample_trg):
    trg_input = sample_trg["trg_input"]
    outputs = decoder(**sample_trg)

    assert outputs.shape == (
        trg_input.shape[0],
        trg_input.shape[1],
        decoder_config.vocab_size,
    )


def test_decoder_forward_bidirectional(decoder_config, sample_trg):
    decoder_config.bidirectional = True
    decoder = DECODERS.build(decoder_config)

    trg_input = sample_trg["trg_input"]

    batch = trg_input.shape[0]
    hidden = torch.zeros(1 * 2, batch, decoder_config.hidden_size)
    cell = torch.zeros_like(hidden)
    hidden_cell = (hidden, cell)

    sample_trg["hidden_cell"] = hidden_cell
    outputs = decoder(**sample_trg)
    assert outputs.shape[-1] == decoder_config.vocab_size


def test_decoder_forward_attention(decoder_config, sample_trg):
    decoder_config.attention = True
    decoder = DECODERS.build(decoder_config)

    trg_input = sample_trg["trg_input"]
    outputs = decoder(**sample_trg)

    assert outputs.shape == (
        trg_input.shape[0],
        trg_input.shape[1],
        decoder_config.vocab_size,
    )


def test_decoder_backward(decoder, sample_trg):
    outputs = decoder(**sample_trg)

    loss = outputs.sum()
    loss.backward()

    for p in decoder.parameters():
        assert p.grad is not None

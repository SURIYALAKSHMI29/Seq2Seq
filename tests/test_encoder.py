import torch
import pytest
import logging

from seq2seq.registry import ENCODERS
from seq2seq.schemas import EncoderConfig

from configs.seq2seq_config import ENCODER_CONFIG


logger = logging.getLogger(__name__)


@pytest.fixture
def encoder_config():
    return EncoderConfig(type="lstm", vocab_size=10, embed_size=8, hidden_size=16)


@pytest.fixture
def encoder(encoder_config):
    torch.manual_seed(0)
    return ENCODERS.build(encoder_config)


@pytest.fixture
def sample_src(encoder_config):
    batch = 4
    src_len = 7
    src = torch.randint(0, encoder_config.vocab_size, (batch, src_len))
    src_lengths = torch.tensor([src_len] * batch)
    return src, src_lengths


def test_encoder_make(encoder_config):

    encoder_config.type = "rnn"
    with pytest.raises(Exception) as e:
        ENCODERS.build(encoder_config)
    logger.info("Raised exception: %s", e.value)


def test_encoder_forward(encoder, encoder_config, sample_src):

    src, src_lengths = sample_src
    outputs, hidden_cell = encoder(src, src_lengths)

    ## batch, seq_len, hidden_size
    assert outputs.shape == (src.shape[0], src.shape[1], encoder_config.hidden_size)

    # batch size check
    assert hidden_cell[0].shape[1] == src.shape[0]


def test_encoder_forward_bidirectional(encoder_config, sample_src):
    encoder_config.bidirectional = True
    encoder = ENCODERS.build(encoder_config)

    src, src_lengths = sample_src

    outputs, hidden = encoder(src, src_lengths)

    # hidden size check
    # bidirectional -> 2*hidden
    assert outputs.shape[-1] == 2 * encoder_config.hidden_size


def test_encoder_backward(encoder, sample_src):
    src, src_lengths = sample_src
    outputs, hidden = encoder(src, src_lengths)

    loss = outputs.sum()
    loss.backward()

    for p in encoder.parameters():
        assert p.grad is not None

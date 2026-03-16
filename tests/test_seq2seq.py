import torch
import torch.nn as nn
import pytest

from seq2seq.modules.seq2seq import Seq2Seq
from seq2seq.schemas import EncoderConfig, DecoderConfig
from seq2seq.registry import ENCODERS, DECODERS

from seq2seq.modules.attention import dot_attention
from configs.seq2seq_config import *


@pytest.fixture
def model_setup():
    torch.manual_seed(0)

    src = torch.tensor([[1, 2, 3], [4, 5, PAD], [6, PAD, PAD]])

    trg_input = torch.tensor([[SOS, 7, 6, 5], [SOS, 4, 3, EOS], [SOS, 2, EOS, PAD]])

    trg_output = torch.tensor([[7, 6, 5, EOS], [4, 3, EOS, PAD], [2, EOS, PAD, PAD]])

    src_lengths = (src != PAD).sum(dim=1)
    vocab_size = 10

    encoder_config = EncoderConfig(
        type="lstm",
        vocab_size=vocab_size,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
    )

    decoder_config = DecoderConfig(
        type="lstm",
        vocab_size=vocab_size,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
    )

    model = Seq2Seq(encoder_config, decoder_config)

    return src, vocab_size, trg_input, trg_output, src_lengths, model


def test_forward(model_setup):
    src, vocab_size, trg_input, trg_output, src_lengths, model = model_setup

    output = model(src, trg_input, src_lengths)
    batch_size, trg_len = trg_output.shape

    print("Output\n", output.shape)

    assert output.shape == (batch_size, trg_len, vocab_size), "Output shape mismatch"

    predictions = output.argmax(dim=2)
    print(predictions)
    assert predictions.min() >= 0 and predictions.max() < vocab_size


def test_attention_weights():
    batch_size, src_len, hidden_size = 2, 3, 4

    encoder_outputs = torch.randn(batch_size, src_len, hidden_size)
    hidden = torch.randn(1, batch_size, hidden_size)

    src_lengths = torch.tensor([src_len] * batch_size)

    context, weights = dot_attention(hidden, encoder_outputs, src_lengths)

    assert context.shape == (batch_size, hidden_size)
    assert weights.shape == (batch_size, src_len)

    sums = weights.sum(dim=1)

    assert torch.allclose(
        sums, torch.ones(batch_size)
    ), "Attention weights must sum to 1"


# def trace_backward(node, depth=0):
#     if node is None:
#         return

#     print(" " * depth, node)

#     for next_fn, _ in node.next_functions:
#         trace_backward(next_fn, depth + 2)


def test_backward(model_setup):
    src, vocab_size, trg_input, trg_output, src_lengths, model = model_setup
    output = model(src, trg_input, src_lengths)

    model.zero_grad()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    loss = criterion(output.reshape(-1, vocab_size), trg_output.reshape(-1))

    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No grad for {name}"

    # trace_backward(loss.grad_fn)


def test_loss_without_ignore(model_setup):
    src, vocab_size, trg_input, trg_output, src_lengths, model = model_setup
    output = model(src, trg_input, src_lengths)

    logits = output.reshape(-1, vocab_size)
    targets = trg_output.reshape(-1)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    loss_ignore = criterion(logits, targets)

    mask = targets != PAD

    criterion_manual = nn.CrossEntropyLoss()
    loss_manual = criterion_manual(logits[mask], targets[mask])

    assert torch.allclose(loss_ignore, loss_manual)

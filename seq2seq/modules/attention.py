import torch
import torch.nn as nn
from torch import Tensor


def dot_attention(hidden: Tensor, encoder_outputs: Tensor, src_lengths: Tensor):

    # hidden = (1, batch, hidden)
    # encoder_outputs = (batch, src_len, hidden)

    # hidden = hidden.permute(1, 2, 0)
    hidden = hidden[-1].unsqueeze(2)
    # (batch, hidden, 1)

    batch, max_src_len, _ = encoder_outputs.shape

    scores = torch.bmm(encoder_outputs, hidden).squeeze(2)
    # print("Scores\n", scores)
    # print("\n Applying softmax before masking\n", torch.softmax(scores, dim=1))

    mask = torch.arange(max_src_len).expand(batch, max_src_len) < src_lengths.unsqueeze(
        1
    )

    # print("\nmask\n", mask)

    masked_scores = scores.masked_fill(mask == False, -1e9)
    # print("masked_scores\n", masked_scores)

    weights = torch.softmax(masked_scores, dim=1)  # (batch, src_len)
    # print("After masking", weights.shape, "\n", weights)

    context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
    # print("Context", context.shape)    # (batch, hidden)

    return context, weights

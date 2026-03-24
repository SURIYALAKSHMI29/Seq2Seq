import torch
import torch.nn as nn
from torch import Tensor


def dot_attention(
    query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None, scale: float = 1.0
):

    scores = torch.matmul(query, key) / scale
    # print("scores", scores.shape)   batch, seq_len, seq_len

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    # print("attn weights", attn_weights.shape)  # batch, seq_len, seq_len

    context = torch.matmul(attn_weights, value)
    # print("Context frm scaled dot attn", context.shape)  # batch, seq_len, attn_dim

    return context


class Attention(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()

        self.d_model = d_model

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        self.scale = d_model**0.5

        # self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None):
        # print(
        #     f"Query: {query.shape} | Key: {key.shape} | Value: {value.shape} | Mask: {mask.shape}"
        # )

        # query, key, value: batch, seq_len, embed
        # mask = (batch, 1, src_len)

        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)

        # print(
        #     f"\nAfter projection, \nQuery: {Q.shape} | Key: {K.shape} | Value: {V.shape}"
        # )

        # Q, K, V: batch, seq_len, attn_dim

        # print("Scale", self.scale)
        context = dot_attention(Q, K.transpose(-2, -1), V, mask, self.scale)

        # context = self.out_proj(context)

        return context

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


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int = 2):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        self.scale = self.head_dim**0.5

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None):
        # print(
        #     f"Query: {query.shape} | Key: {key.shape} | Value: {value.shape} | Mask: {mask.shape}"
        # )

        # query, key, value: batch, seq_len, embed
        # mask = (batch, 1, src_len)

        batch = query.shape[0]

        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)

        # print(
        #     f"\nAfter projection, \nQuery: {Q.shape} | Key: {K.shape} | Value: {V.shape}"
        # )

        # Q, K, V: batch, seq_len, attn_dim

        Q = Q.view(
            batch, Q.shape[1], self.num_heads, self.head_dim
        )  # batch, seq, heads, head_dim

        Q = Q.transpose(1, 2)  # batch, heads, seq, head_dim

        K = K.view(batch, K.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        # batch, heads, seq, head_dim

        V = V.view(batch, V.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        # print("Scale", self.scale)
        context = dot_attention(Q, K.transpose(-2, -1), V, mask, self.scale)

        context = context.transpose(1, 2).contiguous()
        context = context.view(batch, context.shape[1], self.d_model)

        context = self.out_proj(context)

        return context

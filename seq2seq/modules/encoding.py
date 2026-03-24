import torch


def positional_encoding(seq_len, embed_dim):
    pe = torch.zeros(seq_len, embed_dim)

    position = torch.arange(0, seq_len).unsqueeze(1)

    freq = torch.exp(
        torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim)
    )

    pe[:, 0::2] = torch.sin(position * freq)  # even indices
    pe[:, 1::2] = torch.cos(position * freq)  # odd indices

    pe = pe.unsqueeze(0)
    return pe

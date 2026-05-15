import hydra
import torch
import torch.nn as nn
from torch import Tensor

from seq2seq.modules.encoder import EncoderBlock
from seq2seq.schemas import BERTConfig


class BERT(nn.Module):

    def __init__(self, config: BERTConfig):
        super().__init__()

        self.config = config
        self.token_embedding = nn.Embedding(
            config.data.vocab_size, config.embedding.dim
        )
        self.positional_embedding = nn.Embedding(
            config.data.max_src_len, config.embedding.dim
        )
        self.segment_embedding = nn.Embedding(
            config.data.num_segments, config.embedding.dim
        )

        self.activation = self.get_activation(config.attention.activation.lower())

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    config.embedding.dim,
                    config.attention.num_heads,
                    config.regularization.attn_dropout,
                    config.regularization.ffn_dropout,
                    config.attention.ffn_multiplier,
                    self.activation,
                )
                for _ in range(config.attention.layers)
            ]
        )

        self.mlm_head = nn.Linear(config.embedding.dim, config.data.vocab_size)

        self.embed_dropout = nn.Dropout(config.regularization.embed_dropout)
        self.embed_norm = nn.LayerNorm(config.embedding.dim)

        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.constant_(module.weight, 0.02)
            # print(module, "embedding")

        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            # print(module, "linear")

    def get_activation(self, activation_name):
        ACTIVATIONS = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }

        if activation_name not in ACTIVATIONS:
            raise ValueError(
                f"Activation {activation_name} not in {list(ACTIVATIONS.keys())}"
            )

        return ACTIVATIONS[activation_name]

    def embeddings(self, input_ids: Tensor, token_type_ids: Tensor):

        print("X shape", input_ids.shape)  # batch, seq_len

        token_embeddings = self.token_embedding(input_ids)  # batch, seq_len, embed
        print("token embeddings", token_embeddings.shape)

        position_embeddings = self.positional_embedding(
            torch.arange(input_ids.shape[1])
        ).unsqueeze(
            0
        )  # 1, seq_len, embed

        print("x shape[1]", input_ids.shape[1])  # seq_len
        print("position embeddings", position_embeddings.shape)

        print("segments", token_type_ids.shape)  # batch, seq_len
        segment_embeddings = self.segment_embedding(
            token_type_ids
        )  # batch, seq_len, embed
        print("segment embeddings", segment_embeddings.shape)

        norm_embedding = self.embed_norm(
            token_embeddings + position_embeddings + segment_embeddings
        )  # batch, seq_len, embed

        # print("norm embedding", norm_embedding.shape)

        return self.embed_dropout(norm_embedding)  # batch, seq_len, embed

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, input_lengths: Tensor):
        batch, max_src_len = input_ids.shape
        # print("input_ids shape ", input_ids.shape)  # batch, seq_len

        # print("lengths", lengths.shape)  # batch

        x = self.embeddings(input_ids, token_type_ids)

        mask = (
            (
                torch.arange(max_src_len).expand(batch, max_src_len)
                < input_lengths.unsqueeze(1)
            )
            .unsqueeze(1)
            .unsqueeze(1)
        ).to(x.device)

        for layer in self.layers:
            x = layer(x, mask)

        last_hidden_state = x
        cls_output = x[:, 0]

        logits = self.mlm_head(last_hidden_state)

        return logits, last_hidden_state, cls_output


@hydra.main(version_base=None, config_path="../../configs", config_name="bert")
def test_bert(cfg: BERTConfig):

    input_ids = torch.randint(
        1, cfg.data.vocab_size, (cfg.data.batch, cfg.data.max_src_len)
    )
    token_type_ids = torch.randint(0, 2, (cfg.data.batch, cfg.data.max_src_len))
    src_lengths = torch.full_like(input_ids[:, 0], cfg.data.max_src_len)

    print("Config", cfg)
    bert = BERT(cfg)
    print("\n\nBert model", bert)

    last_hidden_state, cls_output = bert(input_ids, token_type_ids, src_lengths)
    print("\n\last_hiden_state shape", last_hidden_state.shape)  # batch, seq_len, embed
    print("cls output shape", cls_output.shape)


if __name__ == "__main__":
    test_bert()

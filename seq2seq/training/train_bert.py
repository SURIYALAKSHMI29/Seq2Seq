import torch
import torch.nn as nn
import torch.optim as optim

import hydra

from seq2seq.schemas import BERTConfig
from seq2seq.modules.bert import BERT


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="bert",
)
def train_bert(cfg: BERTConfig):
    BATCH = cfg.data.batch
    SEQ_LEN = cfg.data.max_src_len
    VOCAB_SIZE = cfg.data.vocab_size

    MASK_TOKEN_ID = VOCAB_SIZE - 1

    input_ids = torch.randint(
        low=1,
        high=MASK_TOKEN_ID,
        size=(BATCH, SEQ_LEN),
    )

    print(input_ids)
    print(f"input_ids shape : {input_ids.shape}\n")  # batch, seq_len

    token_type_ids = torch.randint(
        low=0,
        high=2,
        size=(BATCH, SEQ_LEN),
    )

    print(token_type_ids)
    print(f"token_type_ids shape : {token_type_ids.shape}\n")  # batch, seq_len

    src_lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)

    print(src_lengths)
    print(f"src_lengths shape : {src_lengths.shape}")  # batch

    labels = input_ids.clone()

    print(labels)
    print(f"labels shape : {labels.shape}\n")  # batch, seq_len

    probability_matrix = torch.rand(input_ids.shape)
    mask_positions = probability_matrix < 0.15

    print(mask_positions)
    print(f"mask_positions shape : {mask_positions.shape}\n")  # batch, seq_len

    masked_input_ids = input_ids.clone()
    masked_input_ids[mask_positions] = MASK_TOKEN_ID

    print(masked_input_ids)
    print(f"masked_input_ids shape : {masked_input_ids.shape}\n")  # batch, seq_len

    labels[~mask_positions] = -100

    print(labels)
    print(f"labels shape : {labels.shape}")  # batch, seq_len

    model = BERT(cfg)
    print("\n\nBert model", model)

    logits, _, _ = model(
        masked_input_ids,
        token_type_ids,
        src_lengths,
    )

    # print(logits)
    # print(f"logits shape : {logits.shape}")  # batch, seq_len, vocab_size

    logits_reshaped = logits.view(-1, VOCAB_SIZE)

    labels_reshaped = labels.view(-1)

    print(logits_reshaped)
    print(labels_reshaped)

    print("logits reshaped shape :", logits_reshaped.shape)  # batch * seq, vocab
    print("labels reshaped shape :", labels_reshaped.shape)  # batch * seq

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    loss = criterion(
        logits_reshaped,
        labels_reshaped,
    )

    print("Loss :", loss.item())

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.optimizer.kwargs.lr,
        betas=cfg.optimizer.kwargs.betas,
        eps=cfg.optimizer.kwargs.eps,
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


if __name__ == "__main__":
    train_bert()

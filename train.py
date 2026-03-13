import torch
import torch.nn as nn
import time

from seq2seq.data import get_dataloader
from seq2seq.modules.seq2seq import Seq2Seq
from configs.seq2seq_config import ENCODER_CONFIG, DECODER_CONFIG, TRAIN_CONFIG, PAD
from seq2seq.schemas import EncoderConfig, DecoderConfig


def train():
    (src_vocab, trg_vocab), train_dataloader, _ = get_dataloader()

    ENCODER_CONFIG["vocab_size"] = len(src_vocab)
    DECODER_CONFIG["vocab_size"] = len(trg_vocab)

    print("src vocab size", len(src_vocab))  # 30394
    print("trg vocab size", len(trg_vocab))  # 46715

    encoder_config = EncoderConfig(**ENCODER_CONFIG)
    decoder_config = DecoderConfig(**DECODER_CONFIG)

    model = Seq2Seq(encoder_config, decoder_config)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG["LR"])

    for epoch in range(TRAIN_CONFIG["EPOCHS"]):
        start = time.time()
        model.train()
        total_loss = 0
        for src, trg_input, trg_output in train_dataloader:
            src_lengths = (src != PAD).sum(dim=1)
            optimizer.zero_grad()
            output = model(src, trg_input, src_lengths)
            output_flat = output.view(-1, output.size(-1))
            trg_flat = trg_output.reshape(-1)

            # print("trg_ouput sahpe:", trg_output.shape)
            # print("Flattened trg shape:", trg_flat.shape)
            # print("Flattened output shape:", output_flat.shape)

            loss = criterion(output_flat, trg_flat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{TRAIN_CONFIG['EPOCHS']}, Loss: {total_loss/len(train_dataloader):.4f}"
        )

    print("Time taken", time.time() - start)


if __name__ == "__main__":
    train()

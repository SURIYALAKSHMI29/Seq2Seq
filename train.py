import torch
import torch.nn as nn

from seq2seq.data_loader import get_dataloader
from seq2seq.modules.seq2seq import Seq2Seq
from configs.seq2seq_config import ENCODER_CONFIG, DECODER_CONFIG, TRAIN_CONFIG, PAD
from seq2seq.schemas import EncoderConfig, DecoderConfig


def train():
    train_dataloader = get_dataloader()

    encoder_config = EncoderConfig(**ENCODER_CONFIG)
    decoder_config = DecoderConfig(**DECODER_CONFIG)

    model = Seq2Seq(encoder_config, decoder_config)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG["LR"])

    for epoch in range(TRAIN_CONFIG["EPOCHS"]):
        model.train()
        total_loss = 0
        for src, trg_input, trg_output in train_dataloader:
            src_lengths = (src != PAD).sum(dim=1)
            optimizer.zero_grad()
            output = model(src, trg_input, src_lengths)
            output_flat = output.view(-1, output.size(-1))
            trg_flat = trg_output.view(-1)

            loss = criterion(output_flat, trg_flat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{TRAIN_CONFIG['EPOCHS']}, Loss: {total_loss/len(train_dataloader):.4f}"
        )


if __name__ == "__main__":
    train()

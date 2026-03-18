import torch
from torch.utils.data import DataLoader
from torch.nn.modules import loss
import torch.optim as optim

from seq2seq import Seq2Seq
from configs.seq2seq_config import PAD, EOS


def train_epoch(
    model: Seq2Seq, criterion: loss, optimizer: optim, dataloader: DataLoader
):
    model.train()
    total_loss = 0
    for src, trg_input, trg_output in dataloader:
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

    return total_loss / len(dataloader)


def val_epoch(model: Seq2Seq, criterion: loss, dataloader: DataLoader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, trg_input, trg_output in dataloader:
            src_lengths = (src != PAD).sum(dim=1)

            output = model(src, trg_input, src_lengths, teacher_forcing=False)
            output_flat = output.view(-1, output.size(-1))
            trg_flat = trg_output.reshape(-1)

            loss = criterion(output_flat, trg_flat)

            total_loss += loss.item()

    return total_loss / len(dataloader)

import torch
from torch.utils.data import DataLoader
from torch.nn.modules import loss
import torch.optim as optim
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from seq2seq import Seq2Seq
from configs.seq2seq_config import PAD


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


def evaluate(model, criterion, dataloader, trg_vocab_inv):
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct_tokens = 0
    total_bleu = 0
    smooth = SmoothingFunction().method1  # for BLEU
    samples = []

    with torch.no_grad():
        for src, trg_input, trg_output in dataloader:
            src_lengths = (src != PAD).sum(dim=1)
            output = model(src, trg_input, src_lengths, teacher_forcing=False)
            output_flat = output.view(-1, output.size(-1))
            trg_flat = trg_output.view(-1)

            loss = criterion(output_flat, trg_flat)
            total_loss += loss.item()

            preds = output.argmax(-1)

            mask = trg_output != PAD  # ignore pad tokens
            correct_tokens += (preds == trg_output).masked_select(mask).sum().item()
            total_tokens += mask.sum().item()

            # BLEU score (sentence-level)
            for i in range(src.size(0)):
                # ref = [trg_tokenizer.decode(trg_output[i].tolist()).split()]
                # pred = trg_tokenizer.decode(preds[i].tolist()).split()

                trg_ids = trg_output[i].tolist()
                pred_ids = preds[i].tolist()

                ref_words = [
                    trg_vocab_inv[id]
                    for id in trg_ids
                    if id != PAD and id != trg_vocab_inv.get("<EOS>", -1)
                ]
                pred_words = [
                    trg_vocab_inv[id]
                    for id in pred_ids
                    if id != PAD and id != trg_vocab_inv.get("<EOS>", -1)
                ]

                # BLEU expects reference as a list of references
                ref = [ref_words]
                pred = pred_words

                total_bleu += sentence_bleu(ref, pred, smoothing_function=smooth)

            samples.append((src[0], trg_output[0], preds[0]))

    avg_loss = total_loss / len(dataloader)
    token_acc = correct_tokens / total_tokens
    avg_bleu = total_bleu / len(dataloader.dataset)

    return avg_loss, token_acc, avg_bleu, samples

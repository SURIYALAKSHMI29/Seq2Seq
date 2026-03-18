import torch
import torch.nn as nn

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from seq2seq.data import get_dataloader, get_tokenizers
from seq2seq.modules.seq2seq import Seq2Seq
from configs.seq2seq_config import (
    ENCODER_CONFIG,
    DECODER_CONFIG,
    TRAIN_CONFIG,
    PATHS,
    PAD,
)
from seq2seq.schemas import EncoderConfig, DecoderConfig


def evaluate(model, criterion, dataloader, trg_vocab_inv, EOS):
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

                ref_words = []
                for id in trg_ids:
                    if id == EOS or id == PAD:
                        break
                    ref_words.append(trg_vocab_inv.get(id, "<unk>"))

                pred_words = []
                for id in pred_ids:
                    if id == EOS or id == PAD:
                        break
                    pred_words.append(trg_vocab_inv.get(id, "<unk>"))

                # BLEU expects reference as a list of references
                ref = [ref_words]
                pred = pred_words

                if len(ref_words):
                    total_bleu += sentence_bleu(ref, pred, smoothing_function=smooth)

            samples.append((src[0], trg_output[0], preds[0]))

    avg_loss = total_loss / len(dataloader)
    token_acc = correct_tokens / total_tokens
    avg_bleu = total_bleu / len(dataloader.dataset)

    return avg_loss, token_acc, avg_bleu, samples


def decode_tokens(tensor, vocab, EOS):
    words = []
    # PAD = tokenizer.token_to_id("<PAD>")
    # EOS = tokenizer.token_to_id("<EOS>")
    for idx in tensor:
        if idx.item() == EOS or idx.item() == PAD:
            break
        words.append(vocab.get(idx.item(), -1))
    return " ".join(words)


def validate():
    print("Model path:", PATHS["MODEL"])

    # _, val_loader = get_dataloader()
    (src_vocab, trg_vocab), train_dataloader, val_dataloader = get_dataloader()

    src_vocab_inv = {idx: word for word, idx in src_vocab.items()}
    trg_vocab_inv = {idx: word for word, idx in trg_vocab.items()}

    EOS = trg_vocab["<EOS>"]

    ENCODER_CONFIG["vocab_size"] = len(src_vocab)
    DECODER_CONFIG["vocab_size"] = len(trg_vocab)

    print("src vocab size", len(src_vocab))  # 30394
    print("trg vocab size", len(trg_vocab))  # 46715

    # src_tokenizer, trg_tokenizer = get_tokenizers()

    encoder_config = EncoderConfig(**ENCODER_CONFIG)
    decoder_config = DecoderConfig(**DECODER_CONFIG)

    print("Model instantiated")
    model = Seq2Seq(encoder_config, decoder_config)
    model.load_state_dict(torch.load(PATHS["MODEL"], map_location=torch.device("cpu")))

    # criterion = nn.CrossEntropyLoss(ignore_index=trg_tokenizer.token_to_id("<PAD>"))
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    # path = "/home/suriya-nts0309/seq2seq/trained_models/Seq2Seq_e20_wa.pth"

    avg_loss, token_acc, avg_bleu, samples = evaluate(
        model, criterion, val_dataloader, trg_vocab_inv, EOS
    )

    print(f"Average loss {avg_loss}")
    print(f"Token accuracy {token_acc}")
    print(f"Average BLEU {avg_bleu}")
    print("\nSamples\n")

    for src_tensor, trg_tensor, pred_tensor in samples[:20]:
        print("Input    :", decode_tokens(src_tensor, src_vocab_inv, -1))
        print("Target   :", decode_tokens(trg_tensor, trg_vocab_inv, EOS))
        print("Predicted:", decode_tokens(pred_tensor, trg_vocab_inv, EOS))
        print("\n", "--" * 50, "\n")


if __name__ == "__main__":
    validate()

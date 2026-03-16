import torch
import torch.nn as nn

from seq2seq.data import get_dataloader, get_tokenizers
from seq2seq.modules.seq2seq import Seq2Seq
from configs.seq2seq_config import (
    ENCODER_CONFIG,
    DECODER_CONFIG,
    TRAIN_CONFIG,
    PATHS,
    EOS,
    PAD,
)
from seq2seq.schemas import EncoderConfig, DecoderConfig
from trainer import evaluate


def decode_tokens(tensor, vocab):
    words = []
    # PAD = tokenizer.token_to_id("<PAD>")
    # EOS = tokenizer.token_to_id("<EOS>")
    for idx in tensor:
        if idx.item() == EOS or idx.item() == PAD:
            break
        words.append(vocab.get(idx.item(), -1))
    return " ".join(words)


def validate():
    print("Started training")
    # _, val_loader = get_dataloader()
    (src_vocab, trg_vocab), _, val_dataloader = get_dataloader()

    src_vocab_inv = {idx: word for word, idx in src_vocab.items()}
    trg_vocab_inv = {idx: word for word, idx in trg_vocab.items()}

    ENCODER_CONFIG["vocab_size"] = len(src_vocab)
    DECODER_CONFIG["vocab_size"] = len(trg_vocab)

    print("src vocab size", len(src_vocab))  # 30394
    print("trg vocab size", len(trg_vocab))  # 46715

    # src_tokenizer, trg_tokenizer = get_tokenizers()

    encoder_config = EncoderConfig(**ENCODER_CONFIG)
    decoder_config = DecoderConfig(**DECODER_CONFIG)

    print("Model instantiated")
    model = Seq2Seq(encoder_config, decoder_config)
    model.load_state_dict(torch.load(PATHS["MODEL"]))

    # criterion = nn.CrossEntropyLoss(ignore_index=trg_tokenizer.token_to_id("<PAD>"))
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    # path = "/home/suriya-nts0309/seq2seq/trained_models/Seq2Seq_e20_wa.pth"

    avg_loss, token_acc, avg_bleu, samples = evaluate(
        model, criterion, val_dataloader, trg_vocab_inv
    )

    print(f"Average loss {avg_loss}")
    print(f"Token accuracy {token_acc}")
    print(f"Average BLEU {avg_bleu}")
    print("\nSamples\n")

    for src_tensor, trg_tensor, pred_tensor in samples:
        print("Input    :", decode_tokens(src_tensor, src_vocab_inv))
        print("Target   :", decode_tokens(trg_tensor, trg_vocab_inv))
        print("Predicted:", decode_tokens(pred_tensor, trg_vocab_inv))
        print("\n", "--" * 50, "\n")


if __name__ == "__main__":
    validate()

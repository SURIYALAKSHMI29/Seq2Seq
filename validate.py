import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from seq2seq.data import get_dataloader
from seq2seq.modules.seq2seq import Seq2Seq
from configs.seq2seq_config import ENCODER_CONFIG, DECODER_CONFIG, TRAIN_CONFIG, PAD
from seq2seq.schemas import EncoderConfig, DecoderConfig
from trainer import val_epoch


def validate():
    print("Started training")
    (src_vocab, trg_vocab), _, val_dataloader = get_dataloader()

    ENCODER_CONFIG["vocab_size"] = len(src_vocab)
    DECODER_CONFIG["vocab_size"] = len(trg_vocab)

    print("src vocab size", len(src_vocab))  # 30394
    print("trg vocab size", len(trg_vocab))  # 46715

    encoder_config = EncoderConfig(**ENCODER_CONFIG)
    decoder_config = DecoderConfig(**DECODER_CONFIG)

    print("Model instantiated")
    model = Seq2Seq(encoder_config, decoder_config)
    model = model.load_state_dict(torch.load(TRAIN_CONFIG["PATH"]))

    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    # path = "/home/suriya-nts0309/seq2seq/trained_models/Seq2Seq_e20_wa.pth"

    for epoch in range(TRAIN_CONFIG["EPOCHS"]):
        val_loss = val_epoch(model, criterion, val_dataloader)

        print(f"Epoch {epoch}  |  Val {val_loss}")

    # torch.save(model.state_dict(), path)


if __name__ == "__main__":
    validate()

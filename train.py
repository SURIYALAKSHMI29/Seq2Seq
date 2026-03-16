import torch
import torch.nn as nn
import time
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

from seq2seq.data import get_dataloader
from seq2seq.modules.seq2seq import Seq2Seq
from configs.seq2seq_config import ENCODER_CONFIG, DECODER_CONFIG, TRAIN_CONFIG, PAD
from seq2seq.schemas import EncoderConfig, DecoderConfig
from trainer import train_epoch, val_epoch


def log_params(writer: SummaryWriter, ENCODER_CONFIG, DECODER_CONFIG):
    writer.add_text(
        "Config",
        str(
            {
                "src_vocab_size": ENCODER_CONFIG["vocab_size"],
                "trg_vocab_size": DECODER_CONFIG["vocab_size"],
                "encoder_type": ENCODER_CONFIG["type"],
                "decoder_type": DECODER_CONFIG["type"],
                "encoder_hidden_size": ENCODER_CONFIG["hidden_size"],
                "decoder_hidden_size": DECODER_CONFIG["hidden_size"],
                "encoder_embed_size": ENCODER_CONFIG["embed_size"],
                "decoder_embed_size": DECODER_CONFIG["embed_size"],
                "attention": DECODER_CONFIG["attention"],
                "encoder_bidirectional": ENCODER_CONFIG["bidirectional"],
                "decoder_bidirectional": DECODER_CONFIG["bidirectional"],
                "encoder_layers": ENCODER_CONFIG["layers"],
                "decoder_layers": DECODER_CONFIG["layers"],
                "lr": TRAIN_CONFIG["LR"],
                "batch": TRAIN_CONFIG["BATCH_SIZE"],
                "epochs": TRAIN_CONFIG["EPOCHS"],
            }
        ),
    )


def train():
    print("Started training")
    # train_dataloader, val_dataloader = get_dataloader()
    (src_vocab, trg_vocab), train_dataloader, val_dataloader = get_dataloader()

    ENCODER_CONFIG["vocab_size"] = len(src_vocab)
    DECODER_CONFIG["vocab_size"] = len(trg_vocab)

    print("src vocab size", len(src_vocab))  # 30394
    print("trg vocab size", len(trg_vocab))  # 46715

    encoder_config = EncoderConfig(**ENCODER_CONFIG)
    decoder_config = DecoderConfig(**DECODER_CONFIG)

    run_name = f"attention_{DECODER_CONFIG['attention']}_lr{TRAIN_CONFIG['LR']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/seq2seq_experiments/{run_name}")
    log_params(writer, ENCODER_CONFIG, DECODER_CONFIG)

    print("Model instantiated")
    model = Seq2Seq(encoder_config, decoder_config)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG["LR"])

    path = "/home/suriya-nts0309/seq2seq/trained_models/seq2seq_bpe_5k_lr0.0001.pth"

    start = time.time()
    for epoch in range(TRAIN_CONFIG["EPOCHS"]):
        train_loss = train_epoch(model, criterion, optimizer, train_dataloader)
        val_loss = val_epoch(model, criterion, val_dataloader)

        print(f"Epoch {epoch}  |  Train {train_loss}  |  Val {val_loss}")
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)

    print("Time taken", time.time() - start)
    writer.close()
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    train()

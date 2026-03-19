import torch
import torch.nn as nn
import time
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import hydra
from hydra.utils import instantiate

from datetime import datetime

from seq2seq.data import get_dataloader
from seq2seq.modules.seq2seq import Seq2Seq
from seq2seq.training.trainer import train_epoch, val_epoch

from seq2seq.schemas_hydra import Config

# from configs.seq2seq_config import ENCODER_CONFIG, DECODER_CONFIG, TRAIN_CONFIG, PAD
# from seq2seq.schemas import EncoderConfig, DecoderConfig


def log_params(writer: SummaryWriter, ENCODER_CONFIG, DECODER_CONFIG, train_config):
    writer.add_text(
        "Config",
        str(
            {
                "src_vocab_size": ENCODER_CONFIG["vocab_size"],
                "trg_vocab_size": DECODER_CONFIG["vocab_size"],
                "encoder_type": ENCODER_CONFIG["model_name"],
                "decoder_type": DECODER_CONFIG["model_name"],
                "encoder_hidden_size": ENCODER_CONFIG["hidden_size"],
                "decoder_hidden_size": DECODER_CONFIG["hidden_size"],
                "encoder_embed_size": ENCODER_CONFIG["embed_size"],
                "decoder_embed_size": DECODER_CONFIG["embed_size"],
                "attention": DECODER_CONFIG["attention"],
                "encoder_bidirectional": ENCODER_CONFIG["bidirectional"],
                "decoder_bidirectional": DECODER_CONFIG["bidirectional"],
                "encoder_layers": ENCODER_CONFIG["layers"],
                "decoder_layers": DECODER_CONFIG["layers"],
                "lr": train_config.lr,
                "batch": train_config.batch_size,
                "epochs": train_config.epochs,
            }
        ),
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="seq2seq_main")
def train(cfg: Config):
    train_config = instantiate(cfg.train)
    paths_config = instantiate(cfg.paths)

    print("Started training")
    # train_dataloader, val_dataloader = get_dataloader()
    (src_vocab, trg_vocab), train_dataloader, val_dataloader = get_dataloader(
        train_config, paths_config
    )

    # ENCODER_CONFIG["vocab_size"] = len(src_vocab)
    # DECODER_CONFIG["vocab_size"] = len(trg_vocab)
    cfg.encoder.vocab_size = len(src_vocab)
    cfg.decoder.vocab_size = len(trg_vocab)

    print("src vocab size", len(src_vocab))  # 30394  ## 21573
    print("trg vocab size", len(trg_vocab))  # 46715  ## 35434

    # encoder_config = EncoderConfig(**ENCODER_CONFIG)
    # decoder_config = DecoderConfig(**DECODER_CONFIG)
    encoder_config = instantiate(cfg.encoder)
    decoder_config = instantiate(cfg.decoder)

    run_name = f"attention_{decoder_config.attention}_lr{train_config.lr}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/seq2seq_experiments/{run_name}")
    log_params(writer, encoder_config, decoder_config, train_config)

    model = Seq2Seq(encoder_config, decoder_config)
    print("Model instantiated")
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.PAD)
    optimizer = optim.Adam(model.parameters(), lr=train_config.lr)

    path = "/home/suriya-nts0309/seq2seq/trained_models/seq2seq_en2tam_ep10_lr0.001_b64_dp12_emb256_h512_tf0.6.pth"

    start = time.time()
    for epoch in range(train_config.epochs):
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

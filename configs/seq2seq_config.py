EMBED_SIZE = 256
HIDDEN_SIZE = 128

SRC_VOCAB_SIZE = 2000
TRG_VOCAB_SIZE = 2000

PAD = 0
SOS = 1
EOS = 9


TRAIN_CONFIG = {"BATCH_SIZE": 32, "LR": 0.0001, "EPOCHS": 10}
PATHS = {
    "TRAIN": "data/processed/train_10k_data.pt",
    "VAL": "data/processed/val_10k_data.pt",
    "SRC_VOCAB": "data/processed/src_vocab_10k_data.pkl",
    "TRG_VOCAB": "data/processed/trg_vocab_10k_data.pkl",
    "MODEL": "trained_models/seq2seq_v1.pth",
}

ENCODER_CONFIG = {
    "type": "lstm",
    "vocab_size": SRC_VOCAB_SIZE,
    "embed_size": EMBED_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "layers": 1,
    "bidirectional": True,
}

DECODER_CONFIG = {
    "type": "lstm",
    "vocab_size": TRG_VOCAB_SIZE,
    "embed_size": EMBED_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "layers": 1,
    "bidirectional": False,
    "attention": True,
}

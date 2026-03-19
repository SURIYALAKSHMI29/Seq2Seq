EMBED_SIZE = 256
HIDDEN_SIZE = 512

SRC_VOCAB_SIZE = 2000
TRG_VOCAB_SIZE = 2000

PAD = 0
SOS = 1
EOS = 9

MAX_LEN = 10
NUM_PREFIXES = 10
## Avg target length = 8.72
## Max target length = 55
## Min target length = 3


TRAIN_CONFIG = {"BATCH_SIZE": 64, "LR": 0.001, "EPOCHS": 5}
PATHS = {
    "TRAIN": "data/processed/ef_train_flt.pt",
    "VAL": "data/processed/ef_val_flt.pt",
    "SRC_VOCAB": "data/processed/ef_src_vocab_flt.pkl",
    "TRG_VOCAB": "data/processed/ef_trg_vocab_flt.pkl",
    "MODEL": "trained_models/seq2seq_ef_ep10_lr0.001_b64_dp12_emb256_h512_tf0.6_flt2.pth",
    "RAW_DATA": "data/raw/eng-fra.txt",
}

ENCODER_CONFIG = {
    "category": "encoder",
    "model_name": "lstm",
    "vocab_size": SRC_VOCAB_SIZE,
    "embed_size": EMBED_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "layers": 1,
    "bidirectional": True,
}

DECODER_CONFIG = {
    "category": "decoder",
    "model_name": "lstm",
    "vocab_size": TRG_VOCAB_SIZE,
    "embed_size": EMBED_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "layers": 1,
    "bidirectional": False,
    "attention": True,
}

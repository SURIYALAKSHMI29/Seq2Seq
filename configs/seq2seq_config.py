VOCAB_SIZE = 10
EMBED_SIZE = 7
HIDDEN_SIZE = 5
NUM_OF_SAMPLES = 300
MAX_SRC_LEN = 20

PAD = 0
SOS = 1
EOS = 9


TRAIN_CONFIG = {"BATCH_SIZE": 2, "LR": 0.001, "EPOCHS": 10}

ENCODER_CONFIG = {
    "type": "lstm",
    "vocab_size": VOCAB_SIZE,
    "embed_size": EMBED_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "layers": 1,
    "bidirectional": False,
}

DECODER_CONFIG = {
    "type": "lstm",
    "vocab_size": VOCAB_SIZE,
    "embed_size": EMBED_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "layers": 1,
    "bidirectional": False,
}

EMBED_SIZE = 300
HIDDEN_SIZE = 256

PAD = 0
SOS = 1
EOS = 9


TRAIN_CONFIG = {"BATCH_SIZE": 32, "LR": 0.001, "EPOCHS": 5}

ENCODER_CONFIG = {
    "type": "lstm",
    "embed_size": EMBED_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "layers": 1,
    "bidirectional": False,
}

DECODER_CONFIG = {
    "type": "lstm",
    "embed_size": EMBED_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "layers": 1,
    "bidirectional": False,
}

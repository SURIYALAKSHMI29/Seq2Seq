import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle

from .load_dataset import load_colloquial_tamil_dataset
from .preprocessing import preprocess_dataset, build_vocab, build_tokenizer
from configs.seq2seq_config import TRAIN_CONFIG, PATHS, SRC_VOCAB_SIZE, TRG_VOCAB_SIZE


def build_and_save_dataset():
    train, test = load_colloquial_tamil_dataset()

    src_tokenizer = build_tokenizer(
        train["input"], vocab_size=SRC_VOCAB_SIZE, spl_tokens=["<PAD>", "<UNK>"]
    )
    trg_tokenizer = build_tokenizer(
        train["output"],
        vocab_size=TRG_VOCAB_SIZE,
        spl_tokens=["<PAD>", "<UNk>", "<SOS>", "<EOS>"],
    )

    src, trg_in, trg_out = preprocess_dataset(train, src_tokenizer, trg_tokenizer)
    src_test, trg_in_test, trg_out_test = preprocess_dataset(
        test, src_tokenizer, trg_tokenizer
    )

    with open("data/processed/src_tokenizer_5k_data.pkl", "wb") as f:
        pickle.dump(src_tokenizer, f)

    with open("data/processed/trg_tokenizer_5k_data.pkl", "wb") as f:
        pickle.dump(trg_tokenizer, f)

    torch.save(((src, trg_in, trg_out)), "data/processed/train_5k_bpe_data.pt")
    torch.save(
        (src_test, trg_in_test, trg_out_test), "data/processed/val_5k_bpe_data.pt"
    )

    print("Dataset built and saved")


def get_dataloader():

    batch_size = TRAIN_CONFIG["BATCH_SIZE"]

    print(batch_size)

    train = torch.load(PATHS["TRAIN"])
    val = torch.load(PATHS["VAL"])
    src_train, trg_in_train, trg_out_train = train
    src_test, trg_in_val, trg_out_val = val

    train_dataset = TensorDataset(src_train, trg_in_train, trg_out_train)
    val_dataset = TensorDataset(src_test, trg_in_val, trg_out_val)

    src_vocab = pickle.load(open(PATHS["SRC_VOCAB"], "rb"))
    trg_vocab = pickle.load(open(PATHS["TRG_VOCAB"], "rb"))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return (src_vocab, trg_vocab), train_loader, val_loader
    # return train_loader, val_loader


def get_tokenizers():
    src_tokenizer = pickle.load(open(PATHS["SRC_VOCAB"], "rb"))
    trg_tokenizer = pickle.load(open(PATHS["TRG_VOCAB"], "rb"))
    return src_tokenizer, trg_tokenizer

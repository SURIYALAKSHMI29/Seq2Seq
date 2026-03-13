import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle

from .load_dataset import load_colloquial_tamil_dataset
from .preprocessing import preprocess_dataset, build_vocab
from configs.seq2seq_config import TRAIN_CONFIG


def build_and_save_dataset():
    train, test = load_colloquial_tamil_dataset()

    src_vocab = build_vocab(train["input"])
    trg_vocab = build_vocab(train["output"])

    src, trg_in, trg_out = preprocess_dataset(train, src_vocab, trg_vocab)
    src_test, trg_in_test, trg_out_test = preprocess_dataset(test, src_vocab, trg_vocab)

    with open("data/processed/src_vocab_5k_data.pkl", "wb") as f:
        pickle.dump(src_vocab, f)

    with open("data/processed/trg_vocab_5k_data.pkl", "wb") as f:
        pickle.dump(trg_vocab, f)

    torch.save(((src, trg_in, trg_out)), "data/processed/train_5k_data.pt")
    torch.save((src_test, trg_in_test, trg_out_test), "data/processed/test_5k_data.pt")

    print("Dataset built and saved")


def get_dataloader():

    batch_size = TRAIN_CONFIG["BATCH_SIZE"]

    print(batch_size)

    train = torch.load("data/processed/train_5k_data.pt")
    test = torch.load("data/processed/test_5k_data.pt")
    src_train, trg_in_train, trg_out_train = train
    src_test, trg_in_test, trg_out_test = test

    train_dataset = TensorDataset(src_train, trg_in_train, trg_out_train)
    test_dataset = TensorDataset(src_test, trg_in_test, trg_out_test)

    src_vocab = pickle.load(open("data/processed/src_vocab_5k_data.pkl", "rb"))
    trg_vocab = pickle.load(open("data/processed/trg_vocab_5k_data.pkl", "rb"))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (src_vocab, trg_vocab), train_loader, test_loader

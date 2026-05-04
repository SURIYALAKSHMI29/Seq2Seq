import pickle

import hydra
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader, TensorDataset

from seq2seq.data.load_dataset import load_ceng2french_dataset
from seq2seq.data.preprocessing import build_tokenizer, build_vocab, preprocess_dataset
from seq2seq.schemas import Config

# from configs.seq2seq_config import TRAIN_CONFIG, PATHS, SRC_VOCAB_SIZE, TRG_VOCAB_SIZE


@hydra.main(version_base=None, config_path="../../configs", config_name="seq2seq_main")
def build_and_save_dataset(cfg: Config):
    data_config = instantiate(cfg.data)
    MAX_LEN, NUM_PREFIXES = data_config.vocab.MAX_LEN, data_config.vocab.NUM_PREFIXES
    SRC_VOCAB_SIZE, TRG_VOCAB_SIZE = (
        data_config.vocab.src_vocab_size,
        data_config.vocab.trg_vocab_size,
    )

    train, test = load_ceng2french_dataset(data_config, MAX_LEN, NUM_PREFIXES)

    src_tokenizer = build_tokenizer(
        train["src"], vocab_size=SRC_VOCAB_SIZE, spl_tokens=["<PAD>", "<UNK>"]
    )
    trg_tokenizer = build_tokenizer(
        train["trg"],
        vocab_size=TRG_VOCAB_SIZE,
        spl_tokens=["<PAD>", "<UNk>", "<SOS>", "<EOS>"],
    )

    # encoded = src_tokenizer.encode("pouvons nous parler de la politique ?")
    # print(encoded.tokens, "\n", encoded.ids)

    # encoded_eng = trg_tokenizer.encode(
    #     "<SOS> i was disappointed with your paper. <EOS>"
    # )
    # print(encoded_eng.tokens, "\n", encoded_eng.ids)

    src, trg_in, trg_out = preprocess_dataset(train, src_tokenizer, trg_tokenizer)
    src_test, trg_in_test, trg_out_test = preprocess_dataset(
        test, src_tokenizer, trg_tokenizer
    )

    # src_lengths = [len(src_tokenizer.encode(s)) for s in train["src"]]
    # trg_lengths = [len(trg_tokenizer.encode(s)) for s in train["trg"]]

    # print(
    #     f"src — mean: {sum(src_lengths)/len(src_lengths):.1f}, max: {max(src_lengths)}"
    # )
    # print(
    #     f"trg — mean: {sum(trg_lengths)/len(trg_lengths):.1f}, max: {max(trg_lengths)}"
    # )
    src_vocab = build_vocab(train["src"])
    trg_vocab = build_vocab(train["trg"])

    # src, trg_in, trg_out = preprocess_dataset(train, src_vocab, trg_vocab)
    # src_test, trg_in_test, trg_out_test = preprocess_dataset(test, src_vocab, trg_vocab)

    # with open("data/processed/fe_src_vocab_bpe_1426.pkl", "wb") as f:
    #     pickle.dump(src_tokenizer, f)

    # with open("data/processed/fe_trg_vocab_bpe_1426.pkl", "wb") as f:
    #     pickle.dump(trg_tokenizer, f)

    # torch.save(((src, trg_in, trg_out)), "data/processed/fe_train_bpe_1426.pt")
    # torch.save(
    #     (src_test, trg_in_test, trg_out_test), "data/processed/fe_val_bpe_1426.pt"
    # )

    print("Dataset built and saved")


def get_dataloader(train_config, paths_config):

    batch_size = train_config.batch_size

    print(batch_size)

    train = torch.load(paths_config.train)
    val = torch.load(paths_config.val)
    src_train, trg_in_train, trg_out_train = train
    src_test, trg_in_val, trg_out_val = val

    train_dataset = TensorDataset(src_train, trg_in_train, trg_out_train)
    val_dataset = TensorDataset(src_test, trg_in_val, trg_out_val)

    src_vocab = pickle.load(open(paths_config.src_vocab, "rb"))
    trg_vocab = pickle.load(open(paths_config.trg_vocab, "rb"))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return (src_vocab, trg_vocab), train_loader, val_loader
    # return train_loader, val_loader


def get_tokenizers(paths_config):
    src_tokenizer = pickle.load(open(paths_config.src_vocab, "rb"))
    trg_tokenizer = pickle.load(open(paths_config.trg_vocab, "rb"))
    return src_tokenizer, trg_tokenizer
    return src_tokenizer, trg_tokenizer

import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
import hydra
from hydra.utils import instantiate

from seq2seq.data.load_dataset import load_ceng2french_dataset
from seq2seq.data.preprocessing import preprocess_dataset, build_vocab, build_tokenizer
from seq2seq.schemas_hydra import Config

# from configs.seq2seq_config import TRAIN_CONFIG, PATHS, SRC_VOCAB_SIZE, TRG_VOCAB_SIZE


@hydra.main(version_base=None, config_path="../../configs", config_name="seq2seq_main")
def build_and_save_dataset(cfg: Config):
    paths_config = instantiate(cfg.paths)
    MAX_LEN, NUM_PREFIXES = cfg.MAX_LEN, cfg.NUM_PREFIXES

    train, test = load_ceng2french_dataset(paths_config, MAX_LEN, NUM_PREFIXES)

    # src_tokenizer = build_tokenizer(
    #     train["input"], vocab_size=SRC_VOCAB_SIZE, spl_tokens=["<PAD>", "<UNK>"]
    # )
    # trg_tokenizer = build_tokenizer(
    #     train["output"],
    #     vocab_size=TRG_VOCAB_SIZE,
    #     spl_tokens=["<PAD>", "<UNk>", "<SOS>", "<EOS>"],
    # )

    # src, trg_in, trg_out = preprocess_dataset(train, src_tokenizer, trg_tokenizer)
    # src_test, trg_in_test, trg_out_test = preprocess_dataset(
    #     test, src_tokenizer, trg_tokenizer
    # )

    src_vocab = build_vocab(train["src"])
    trg_vocab = build_vocab(train["trg"])

    src, trg_in, trg_out = preprocess_dataset(train, src_vocab, trg_vocab)
    src_test, trg_in_test, trg_out_test = preprocess_dataset(test, src_vocab, trg_vocab)

    with open("data/processed/ef_src_vocab_flt_b32.pkl", "wb") as f:
        pickle.dump(src_vocab, f)

    with open("data/processed/ef_trg_vocab_flt_b32.pkl", "wb") as f:
        pickle.dump(trg_vocab, f)

    torch.save(((src, trg_in, trg_out)), "data/processed/ef_train_flt_b32.pt")
    torch.save(
        (src_test, trg_in_test, trg_out_test), "data/processed/ef_val_flt_b32.pt"
    )

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

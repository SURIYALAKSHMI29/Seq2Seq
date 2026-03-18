import random
from collections import Counter

from configs.seq2seq_config import PATHS, MAX_LEN, NUM_PREFIXES


def split_pairs(pairs):
    src, trg = [], []
    for s, t in pairs:
        t = "<SOS> " + t + " <EOS>"
        src.append(s)
        trg.append(t)
    return {"src": src, "trg": trg}


def get_common_prefixes(pairs):
    starts = [tuple(s.split()[:2]) for s, _ in pairs if len(s.split()) >= 2]
    counter = Counter(starts)
    most_common = [word for word, _ in counter.most_common(NUM_PREFIXES)]
    return most_common


def filterPair(pair, eng_prefixes):
    if len(pair[0].split(" ")) > MAX_LEN or len(pair[1].split(" ")) > MAX_LEN:
        return False

    first_two = tuple(pair[0].split()[:2])
    return first_two in eng_prefixes


def filterPairs(pairs, eng_prefixes):
    return [pair for pair in pairs if filterPair(pair, eng_prefixes)]


def load_ceng2french_dataset():
    pairs = []
    path = PATHS["RAW_DATA"]
    with open(path, encoding="utf-8") as f:
        for line in f:
            eng, fra = line.strip().lower().split("\t")
            pairs.append((eng, fra))

    print(pairs[:5])
    print("Total number of samples:", len(pairs))

    eng_prefixes = get_common_prefixes(pairs)

    print("eng_prefixes", eng_prefixes)

    pairs = filterPairs(pairs, eng_prefixes)
    print(f"After filtering, total num of pairs {len(pairs)}")

    random.shuffle(pairs)

    split_idx = int(len(pairs) * 0.8)

    train = split_pairs(pairs[:split_idx])
    test = split_pairs(pairs[split_idx:])

    print("train samples", len(train["src"]))
    print("test samples", len(test["src"]))

    return train, test

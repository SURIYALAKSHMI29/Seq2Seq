import random
from collections import Counter


def split_pairs(pairs):
    src, trg = [], []
    for eng, french in pairs:
        eng = "<SOS> " + eng + " <EOS>"
        src.append(french)
        trg.append(eng)
    return {"src": src, "trg": trg}


def get_common_prefixes(pairs, NUM_PREFIXES):
    starts = [tuple(s.split()[:2]) for s, _ in pairs if len(s.split()) >= 2]
    counter = Counter(starts)
    most_common = [word for word, _ in counter.most_common(NUM_PREFIXES)]
    return most_common


def filterPair(pair, prefixes, MAX_LEN):
    if len(pair[0].split(" ")) > MAX_LEN or len(pair[1].split(" ")) > MAX_LEN:
        return False

    first_two = tuple(pair[0].split()[:2])
    return first_two in prefixes


def filterPairs(pairs, prefixes, MAX_LEN):
    return [pair for pair in pairs if filterPair(pair, prefixes, MAX_LEN)]


def load_ceng2french_dataset(paths, MAX_LEN, NUM_PREFIXES):
    pairs = []
    path = paths.raw_data
    with open(path, encoding="utf-8") as f:
        for line in f:
            eng, fra = line.strip().lower().split("\t")
            pairs.append((eng, fra))

    print(pairs[:5])
    print("Total number of samples:", len(pairs))

    prefixes = get_common_prefixes(pairs, NUM_PREFIXES)

    # print("prefixes", prefixes)

    pairs = filterPairs(pairs, prefixes, MAX_LEN)
    print(f"After filtering, total num of pairs {len(pairs)}")

    random.shuffle(pairs)

    split_idx = int(len(pairs) * 0.8)

    train = split_pairs(pairs[:split_idx])
    test = split_pairs(pairs[split_idx:])

    print("train samples", len(train["src"]))
    print("test samples", len(test["src"]))

    return train, test

import torch
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer, models, trainers, pre_tokenizers


def build_vocab(sentences):
    words = [word for sent in sentences for word in sent.strip().split()]
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
    }

    for word in set(words):
        vocab[word] = len(vocab)

    return vocab


## used for bpe
def build_tokenizer(data, vocab_size, spl_tokens):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=spl_tokens)
    tokenizer.train_from_iterator(data, trainer=trainer)
    return tokenizer


def encode_and_pad_bpe(sentences, tokenizer):
    sequences = []
    for sent in sentences:
        seq = tokenizer.encode(sent).ids
        sequences.append(torch.tensor(seq, dtype=torch.long))

    return pad_sequence(
        sequences, batch_first=True, padding_value=tokenizer.token_to_id("<PAD>")
    )


def encode_sentence(sentence, vocab):
    return [vocab.get(word, vocab["<unk>"]) for word in sentence.strip().split()]


def pad_data(sentences, vocab):
    sequences = [encode_sentence(sent, vocab) for sent in sentences]
    # print(sequences[1], sequences[5])

    seq_tensors = [torch.tensor(seq, dtype=torch.long) for seq in sequences]

    padded_seq = pad_sequence(
        seq_tensors, batch_first=True, padding_value=vocab["<pad>"]
    )
    # print(padded_seq[1].shape, padded_seq[5].shape)

    return padded_seq


def preprocess_dataset(dataset, src_tokenizer, trg_tokenizer):
    src_sentences, trg_sentences = dataset["input"], dataset["output"]
    # print(src_sentences[5], trg_sentences[5])

    trg_sentences = [f"<SOS> {sentence} <EOS>" for sentence in trg_sentences]
    # print(trg_sentences[5])

    # src_padded = pad_data(src_sentences, src_vocab)
    # trg_padded = pad_data(trg_sentences, trg_vocab)

    src_padded = encode_and_pad_bpe(src_sentences, src_tokenizer)
    trg_padded = encode_and_pad_bpe(trg_sentences, trg_tokenizer)

    src = src_padded
    trg_input = trg_padded[:, :-1]
    trg_output = trg_padded[:, 1:]

    return src, trg_input, trg_output

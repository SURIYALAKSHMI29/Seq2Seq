import torch
import torch.nn as nn
from torch import Tensor

import hydra
from hydra.utils import instantiate

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu

from seq2seq.data import get_dataloader, get_tokenizers
from seq2seq.modules.seq2seq import Seq2Seq

from seq2seq.schemas import Config


def evaluate(model, criterion, dataloader, trg_vocab_inv, PAD, EOS):
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct_tokens = 0
    smooth = SmoothingFunction().method1  # for BLEU
    samples = []

    with torch.no_grad():
        all_refs = []
        all_preds = []

        for src, trg_input, trg_output in dataloader:
            src_lengths = (src != PAD).sum(dim=1)
            output = model(src, trg_input, src_lengths, teacher_forcing=False)
            output = output[:, : trg_output.size(1), :]

            output_flat = output.reshape(-1, output.size(-1))
            trg_flat = trg_output.view(-1)

            loss = criterion(output_flat, trg_flat)
            total_loss += loss.item()

            preds = output.argmax(-1)

            mask = trg_output != PAD  # ignore pad tokens
            correct_tokens += (preds == trg_output).masked_select(mask).sum().item()
            total_tokens += mask.sum().item()

            # BLEU score (sentence-level)
            for i in range(src.size(0)):
                # ref = [trg_tokenizer.decode(trg_output[i].tolist()).split()]
                # pred = trg_tokenizer.decode(preds[i].tolist()).split()

                # trg_ids = trg_output[i].tolist()
                # pred_ids = preds[i].tolist()

                # ref_words = []
                # for id in trg_ids:
                #     if id == EOS or id == PAD:
                #         break
                #     ref_words.append(trg_vocab_inv.get(id, "<unk>"))

                # pred_words = []
                # for id in pred_ids:
                #     if id == EOS or id == PAD:
                #         break
                #     pred_words.append(trg_vocab_inv.get(id, "<unk>"))

                ref_words = decode_tokens(trg_output[i], trg_vocab_inv)
                pred_words = decode_tokens(preds[i], trg_vocab_inv)

                # BLEU expects reference as a list of references
                ref = [ref_words]
                pred = pred_words

                if len(ref_words):
                    # total_bleu += sentence_bleu(ref, pred, smoothing_function=smooth)
                    all_refs.append(ref)
                    all_preds.append(pred)

            samples.append((src[0], trg_output[0], preds[0]))

    avg_loss = total_loss / len(dataloader)
    token_acc = correct_tokens / total_tokens
    # avg_bleu = total_bleu / len(dataloader.dataset)

    avg_bleu = corpus_bleu(all_refs, all_preds, smoothing_function=smooth)
    return avg_loss, token_acc, avg_bleu, samples


def decode_tokens(data: Tensor, tokenizer, debug=False):
    token_ids = data.tolist()
    num_tokens = len(token_ids)
    if debug:
        if num_tokens != 29:
            print(
                num_tokens,
                tokenizer.decode(token_ids, skip_special_tokens=True),
                token_ids,
            )

        if token_ids.__contains__(tokenizer.token_to_id("<EOS>")):
            print("has <EOS> token")
            print(token_ids.index(tokenizer.token_to_id("<EOS>")))

    if tokenizer.token_to_id("<EOS>") in token_ids:
        eos_indx = token_ids.index(tokenizer.token_to_id("<EOS>"))
        token_ids = token_ids[:eos_indx]

    decoded = tokenizer.decode(token_ids)

    return decoded


# def decode_tokens(data: Tensor, vocab, PAD, EOS):
#     words = []
#     for idx in data:
#         if idx.item() == EOS or idx.item() == PAD:
#             break
#         words.append(vocab.get(idx.item(), -1))
#     return " ".join(words)


@hydra.main(version_base=None, config_path="../../configs", config_name="seq2seq_main")
def validate(cfg: Config):
    print("Model path:", cfg.paths.model)
    train_config = instantiate(cfg.train)
    paths_config = instantiate(cfg.paths)

    # _, val_loader = get_dataloader()
    # (src_vocab, trg_vocab), train_dataloader, val_dataloader = get_dataloader(
    #     train_config, paths_config
    # )

    # src_vocab_inv = {idx: word for word, idx in src_vocab.items()}
    # trg_vocab_inv = {idx: word for word, idx in trg_vocab.items()}

    # EOS = trg_vocab["<EOS>"]
    # PAD = cfg.data.PAD

    (src_vocab_tk, trg_vocab_tk), train_dataloader, val_dataloader = get_dataloader(
        train_config, paths_config
    )

    EOS = trg_vocab_tk.token_to_id("<EOS>")
    PAD = trg_vocab_tk.token_to_id("<PAD>")

    cfg.encoder.vocab_size = src_vocab_tk.get_vocab_size()
    cfg.decoder.vocab_size = trg_vocab_tk.get_vocab_size()

    print("src vocab size", cfg.encoder.vocab_size)  # 6568
    print("trg vocab size", cfg.decoder.vocab_size)  # 4586

    # src_tokenizer, trg_tokenizer = get_tokenizers()

    encoder_config = instantiate(cfg.encoder)
    decoder_config = instantiate(cfg.decoder)

    print("Model instantiated")
    model = Seq2Seq(encoder_config, decoder_config)
    model.load_state_dict(torch.load(cfg.paths.model, map_location=torch.device("cpu")))

    # criterion = nn.CrossEntropyLoss(ignore_index=trg_tokenizer.token_to_id("<PAD>"))
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    # path = "/home/suriya-nts0309/seq2seq/trained_models/Seq2Seq_e20_wa.pth"

    avg_loss, token_acc, avg_bleu, samples = evaluate(
        model, criterion, val_dataloader, trg_vocab_tk, PAD, EOS
    )

    print(f"Average loss {avg_loss}")
    print(f"Token accuracy {token_acc}")
    print(f"Average BLEU {avg_bleu}")
    print("\nSamples\n")

    for src_tensor, trg_tensor, pred_tensor in samples[:20]:
        # print("Input    :", decode_tokens(src_tensor, src_vocab_inv, PAD, -1))
        # print("Target   :", decode_tokens(trg_tensor, trg_vocab_inv, PAD, EOS))
        # print("Predicted:", decode_tokens(pred_tensor, trg_vocab_inv, PAD, EOS))

        print("Input    :", decode_tokens(src_tensor, src_vocab_tk))
        print("Target   :", decode_tokens(trg_tensor, trg_vocab_tk))
        print("Predicted:", decode_tokens(pred_tensor, trg_vocab_tk))

        print("\n", "--" * 50, "\n")


if __name__ == "__main__":
    validate()

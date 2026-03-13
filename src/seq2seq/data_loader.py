# import torch
# from torch.utils.data import Dataset, DataLoader

# from configs.seq2seq_config import *
# from torch.nn.utils.rnn import pad_sequence


# class ReverseNumberDataset(Dataset):
#     """
#     Reverse Number Dataset
#         src -> number   ( 2, 3)
#         trg -> reversed number (3, 2)

#     Here, 1 -> SOS, 9 -> EOS and 0 -> PAD
#     Appending SOS and EOS to target

#     If target length varies, padding is added

#     """

#     def __init__(self, num_samples=NUM_OF_SAMPLES, max_len=MAX_SRC_LEN):
#         self.data = []
#         self.vocab_size = 10

#         self.trg_len = max_len + 2  ## SOS and EOS

#         for _ in range(num_samples):
#             src_len = torch.randint(1, max_len + 1, (1,)).item()

#             src = torch.randint(2, 9, (src_len,))
#             # print("\nSource")
#             # print(src.shape, src)

#             ## concat SOS, src, EOS
#             trg = torch.cat(
#                 [torch.tensor([SOS]), torch.flip(src, dims=(0,)), torch.tensor([EOS])]
#             )

#             if len(trg) < self.trg_len:
#                 trg = torch.cat([trg, torch.full((self.trg_len - len(trg),), PAD)])
#             # print("Target")
#             # print(trg.shape, trg)

#             padded_src = torch.full((max_len,), PAD)
#             padded_src[: len(src)] = src

#             self.data.append((padded_src, trg))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         src, trg = self.data[idx]
#         return src, trg[:-1], trg[1:]


# def get_dataloader():
#     dataset = ReverseNumberDataset()
#     return DataLoader(dataset, batch_size=TRAIN_CONFIG["BATCH_SIZE"], shuffle=True)

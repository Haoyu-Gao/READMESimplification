import torch
from torch import nn

# from utils import *
import json
from torch.utils.data import Dataset
from tokenizers import Tokenizer
import pandas as pd


class Transformer(nn.Module):

    """
    Fine tune -- transfer learning ....
    """
    def __init__(self, model_config, device):
        super(Transformer, self).__init__()
        src_vocab_size = model_config['src_vocab_size']
        trg_vocab_size = model_config['trg_vocab_size']
        embedding_size = model_config['embedding_size']
        max_len = model_config['max_length']
        num_heads = model_config['num_heads']
        num_encoder_layers = model_config['num_encoder_layers']
        num_decoder_layers = model_config['num_decoder_layers']
        ff = model_config['ff']
        dropout = model_config['dropout']

        self.device = device
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(embedding_size, num_heads,
                                            ff, dropout, batch_first=True), num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer=nn.TransformerDecoderLayer(embedding_size, num_heads,
                                            ff, dropout, batch_first=True), num_layers=num_decoder_layers)

        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = model_config['src_pad_idx']

    def make_src_mask(self, src):
        src_mask = src == self.src_pad_idx

        return src_mask

    def forward(self, src, trg):
        N, src_seq_length= src.shape
        N, trg_seq_length = trg.shape

        src_positions = torch.arange(0, src_seq_length).unsqueeze(0).expand(N, src_seq_length).to(self.device)
        trg_positions = torch.arange(0, trg_seq_length).unsqueeze(0).expand(N, trg_seq_length).to(self.device)

        embed_src = self.dropout(self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        embed_trg = self.dropout(self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))

        src_padding_mask = self.make_src_mask(src)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)
        encoded_state = self.encoder(src=embed_src, src_key_padding_mask=src_padding_mask)
        out = self.decoder(tgt=embed_trg, memory=encoded_state, tgt_mask=tgt_mask)

        out = self.fc_out(out)

        return out


class MyDataset(Dataset):

    def __init__(self, file_path_src, file_path_dst, tokenizer):
        self.file_path_src = file_path_src
        self.file_path_dst = file_path_dst
        self.tokenizer = tokenizer
        self.data_src, self.data_dst = [], []
        with open(file_path_src) as f:
            for line in f.readlines():
                self.data_src.append(line)
        with open(file_path_dst) as f:
            for line in f.readlines():
                self.data_dst.append(line)

    def __len__(self):
        return len(self.data_src)

    def __getitem__(self, idx):
        data_src, data_dst = self.data_src[idx], self.data_dst[idx]
        data_src_tensor = torch.tensor(self.tokenizer.encode(data_src).ids)
        data_dst_tensor = torch.tensor(self.tokenizer.encode(data_dst).ids)

        return data_src_tensor, data_dst_tensor


class MyDatasetSingle(Dataset):

    def __init__(self, file_path, tokenizer):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.data = []
        with open(file_path) as f:
            for line in f.readlines():
                self.data.append(line)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data_tensor = torch.tensor(self.tokenizer.encode(data).ids)

        return data_tensor


if __name__ == '__main__':
    pass


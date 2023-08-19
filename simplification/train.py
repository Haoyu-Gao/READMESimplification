import torch
from torch import nn
from torch.utils.data import DataLoader

from tokenizers import Tokenizer
from argparse import ArgumentParser
from utils import *
import json
import os

from copy import deepcopy


from my_model import Transformer, MyDataset


def train(args):
    checkpoint, config, model_config, save_path = args.checkpoint, args.config, args.model, args.save_path
    load_model = False if checkpoint is None else True
    assert config is not None, "Please specify the training config file"
    assert model_config is not None, "Please specify the model configuration"

    training_dict = json.load(open(config))
    model_config = json.load(open(model_config))
    tokenizer = Tokenizer.from_file(training_dict["tokenizer"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.data_source == "wiki":
        train_set = MyDataset("../simplification_data/train.src", "../simplification_data/train.dst", tokenizer)
        dev_set = MyDataset("../simplification_data/valid.src", "../simplification_data/valid.dst", tokenizer)
    else:
        train_set = MyDataset("../md_simplification_data/train.src", "../md_simplification_data/train.dst", tokenizer)
        dev_set = MyDataset("../md_simplification_data/valid.src", "../md_simplification_data/valid.dst", tokenizer)



    step = 0

    model = Transformer(model_config, device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=training_dict['lr'])

    if load_model:
        load_checkpoint(model, optimizer, torch.load(checkpoint))

    train_loader = DataLoader(train_set, batch_size=training_dict['batch_size'])
    dev_loader = DataLoader(dev_set, batch_size=training_dict['batch_size'])

    criterion = nn.CrossEntropyLoss(ignore_index=9)

    best_valid_loss = float('inf')
    best_valid_bleu_score = 0

    for epoch in range(training_dict['num_epochs']):
        print(f"[Epoch {epoch} / {training_dict['num_epochs']}]")

        model.train()
        train_loss = 0
        # train_bleu = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            src_batch, dst_batch = batch[0], batch[1]
            src_batch, dst_batch = src_batch.to(device), dst_batch.to(device)
            output = model(src_batch, dst_batch[:, :-1])
            output = torch.transpose(output, 1, 2)
            target = dst_batch[:, 1:]
            loss = criterion(output, target)
            loss.backward()
            train_loss += loss.item()
            print(src_batch.shape)
            # train_bleu += get_bleu_score(model, tokenizer.decode_batch(src_batch.tolist()),
            #                              tokenizer.decode_batch(dst_batch.tolist()), tokenizer, 50)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()
        print("Training Loss is {}".format(train_loss/len(train_loader)))
        # print("Bleu score on training set is {}".format(train_bleu/len(train_loader)))

        model.eval()
        valid_loss = 0
        valid_bleu = 0
        for batch in dev_loader:
            src_batch, dst_batch = batch[0], batch[1]
            src_batch, dst_batch = src_batch.to(device), dst_batch.to(device)
            output = model(src_batch, dst_batch[:, :-1])
            output = torch.transpose(output, 1, 2)
            target = dst_batch[:, 1:]
            loss = criterion(output, target)
            valid_loss += loss.item()
            valid_bleu += get_bleu_score(model, tokenizer.decode_batch(src_batch.tolist()),
                                         tokenizer.decode_batch(dst_batch.tolist()), deepcopy(tokenizer), 150)

        print("Validation Loss is {}".format(valid_loss/len(dev_loader)))
        print("Bleu score on validation set is {}".format(valid_bleu/len(dev_loader)))

        if valid_bleu > best_valid_bleu_score or valid_loss < best_valid_loss:
            best_valid_bleu_score = valid_bleu
            best_valid_loss = valid_loss
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            save_checkpoint(checkpoint, save_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Specify checkpoint file, None means train from scratch")
    parser.add_argument("--config", type=str, help="Specify training hyperparameter file.")
    parser.add_argument("--model", type=str, help="Specify the model config path")
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--data_source", type=str)
    args = parser.parse_args()
    train(args)



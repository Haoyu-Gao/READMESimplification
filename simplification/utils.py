import os.path

import torch
from pytorch_beam_search import seq2seq
from my_model import Transformer
import json
from tokenizers import Tokenizer
from torchtext.data.metrics import bleu_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state, save_path):
    print("==> Saving checkpoint!")
    os.path.join(save_path, "model")
    torch.save(state, save_path)


def load_checkpoint(model, optimizer, checkpoint):
    print("==> Loading checkpoint!")
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])


def beam_search(model, inputs, tokenizer, max_len, beam_width):
    if type(inputs) is not torch.Tensor:
        encode_texts = tokenizer.encode_batch(inputs)
        encode_texts_tensor = torch.tensor([encode_text.ids for encode_text in encode_texts])

    else:
        encode_texts_tensor = inputs
    # print(encode_texts_tensor.shape)
    predictions, log_probs = seq2seq.beam_search(model, encode_texts_tensor, predictions=max_len, beam_width=beam_width,
                                                 batch_size=len(encode_texts_tensor), progress_bar=1)
    
    predictions1 = predictions[:, 0, :]
    decodes = tokenizer.decode_batch(predictions1.tolist(), skip_special_tokens=False)

    return predictions, log_probs, decodes


def get_bleu_score(model, data, targets, tokenizer, max_len):

    # batch_size = len(data)

    predictions, log_probs, decodes = beam_search(model, data, tokenizer, max_len)

    b = 0
    tokenizer.no_padding()
    for i in range(len(decodes)):

        b = bleu_score([tokenizer.encode(decodes[i], add_special_tokens=False).tokens],
                       [tokenizer.encode(targets[i], add_special_tokens=False).tokens])


    return b / batch_size





import argparse
import json
import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from utils import load_checkpoint, beam_search
from my_model import Transformer, MyDatasetSingle
from postprocessing import postprocessing


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate(args):
    checkpoint = args.model
    file_path = args.path
    beam = args.beam

    tokenizer = Tokenizer.from_file("my_tokenizer.json")
    model_config = json.load(open("model_config.json"))
    model = Transformer(model_config, device)
    model.to(device)

    load_checkpoint(model, None, torch.load(checkpoint, map_location=torch.device('cpu')))

    test_dataset = MyDatasetSingle(file_path, tokenizer)
    loader = DataLoader(test_dataset, batch_size=4)

    decodes = []

    for data in loader:
        predictions, log_probs, decode = beam_search(model, data.to(device), tokenizer, 3, beam)

        for item in decode:
            decodes.append(item)

    with open(args.to_path, 'w') as f:
        for line in decodes:
            f.write(line + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Specify model checkpoint path")
    parser.add_argument("--path", type=str, help="Specify file path")
    parser.add_argument("--beam", type=int, help="Beam search width")
    parser.add_argument("--to_path", type=str, help="Specify written file path")

    args = parser.parse_args()
    generate(args)
    postprocessing(args.path)

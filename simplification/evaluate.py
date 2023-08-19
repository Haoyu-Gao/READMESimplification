import argparse
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from tokenizers import Tokenizer
from nltk.tokenize import word_tokenize

def evaluate(args):
    score = 0

    candidates = []
    references = []
    candidates_path = args.candidate
    references_path = args.reference

    with open(candidates_path) as f:
        for line in f.readlines():
            candidates.append(line)
    with open(references_path) as f:
        for line in f.readlines():
            references.append(line)

    smoothie = SmoothingFunction().method4

    for i in range(len(candidates)):
        candidate = candidates[i]
        reference = references[i]
        score += sentence_bleu([word_tokenize(reference)], word_tokenize(candidate),
                    [0.25, 0.25, 0.25, 0.25], smoothing_function=smoothie)

    score /= len(candidates)
    print("The bleu score is {}".format(score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=str, help="Specify candidate text path")
    parser.add_argument("--reference", type=str, help="Specify reference text path")
    args = parser.parse_args()

    evaluate(args)

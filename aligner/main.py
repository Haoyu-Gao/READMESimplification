import multiprocessing

import nltk
import csv
import json
import pickle
from scipy import spatial

import argparse
from aligner_util import clean_md_text, markdown_to_text, clean_sentences
from model import *


lock = multiprocessing.Lock()

nltk.download("punkt")

vectorizer = pickle.load(open("../data/tfidf.pkl", "rb"))


def identical_or_no_match(simp_sent, norm_sent):
    if simp_sent == norm_sent:
        return True
    tfidf = vectorizer.transform([simp_sent, norm_sent]).todense()
    d = spatial.distance.cosine(tfidf[0, :], tfidf[1, :])
    if d > 0.5:
        return True
    return False


def do_alignment(aligner, record, opath, idx, condition):
    normal = markdown_to_text(record['src'])
    simple = markdown_to_text(record['dest'])

    norm_sents = nltk.sent_tokenize(normal)
    simp_sents = nltk.sent_tokenize(simple)

    aligning_pairs = aligner.get_pairs_with_bert(simp_sents, norm_sents, window_size=25)

    with open(opath, 'a') as f:
        writer = csv.writer(f)

        skip_idx = []
        for pair in aligning_pairs:
            simp_idx, norm_idx = pair
            simp_sent, norm_sent = simp_sents[simp_idx], norm_sents[norm_idx]
            if identical(simp_sent, norm_sent):
                skip_idx.append(simp_idx)

        # if not flag:
        for pair in aligning_pairs:
            simp_idx, norm_idx = pair
            simp_sent, norm_sent = simp_sents[simp_idx], norm_sents[norm_idx]
            simp_sent = simp_sent.replace("\n", " ")
            norm_sent = norm_sent.replace("\n", " ")
            if not identical(simp_sent, norm_sent) and within_bleu_range(simp_sent,
                                                                         norm_sent) and simp_idx not in skip_idx:
                writer.writerow([idx, simp_idx, norm_idx, record['repo_name'], simp_sent, norm_sent])



def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert, do_lower_case=True)
    sequence_classifier = BertForSequenceClassification.from_pretrained(args.bert, output_hidden_states=True)
    aligner = Aligner(sequence_classifier, tokenizer)
    aligner.to(my_device)
    idx = 0

    with open(args.ipath) as f:
        while True:
            line = f.readline()
            if line == "":
                break
            else:
                record = json.loads(line.strip())
                try:
                    do_alignment(aligner, record, args.opath, idx)
                    idx += 1
                except:
                    idx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipath", type=str, help="Specify the input file path")
    parser.add_argument("--bert", type=str, help="Specify the BERT CONFIG and PARAMs folder path")
    parser.add_argument("--opath", type=str, help="Specify the output file path")
    args = parser.parse_args()
    main(args)
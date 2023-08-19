from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import json
import pickle
from aligner_util import *
import re
import nltk
from nltk.stem import WordNetLemmatizer


def main(args):
    ipath, opath = args.ipath, args.opath
    vectorizer = TfidfVectorizer()
    # read in a large chunk of document, and use it for compute Tf-idf
    document_list = []
    idx = 0
    with open(ipath, "r") as f:
        while True:
            line = f.readline()
            if line == "" or len(document_list) == args.docs:
                break
            else:
                print(idx)
                idx += 1
                record = json.loads(line.strip())
                normal_text, simple_text = record['src'], record['dest']
                norm_sents = nltk.sent_tokenize(markdown_to_text(normal_text))
                simp_sents = nltk.sent_tokenize(markdown_to_text(simple_text))
                for norm_sent in norm_sents:
                    if len(nltk.word_tokenize(norm_sent)) > 5:
                        document_list.append(norm_sent)
                for simp_sent in simp_sents:
                    if len(nltk.word_tokenize(simp_sent)) > 5:
                        document_list.append(simp_sent)

    vectorizer.fit_transform(document_list)
    pickle.dump(vectorizer, open(opath, "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipath", type=str, help="Specify the input file path")
    parser.add_argument("--opath", type=str, help="Specify the output tf-idf model path")
    parser.add_argument("--docs", type=int, help="Specify the number of documents to compute out tf-idf vectorizer")

    args = parser.parse_args()
    main(args)

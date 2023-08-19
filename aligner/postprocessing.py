import pandas as pd
import nltk
import csv

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


def remove_outliers(from_path, to_path):
    original_file = pd.read_csv(from_path, header=None)
    count = 0
    prev = None
    cache = []
    with open(to_path, "w") as f:
        writer = csv.writer(f)
        for i in range(len(original_file)):
            if prev != (original_file.loc[i, 0], original_file.iloc[i, 1]):
                if count < 3:
                    for record in cache:
                        writer.writerow(record)
                count = 0
                prev = (original_file.loc[i, 0], original_file.iloc[i, 1])
                cache = []
            count += 1

            simp, norm = original_file.iloc[i, 3], original_file.iloc[i, 4]

            simp_tok, norm_tok = nltk.word_tokenize(simp), nltk.word_tokenize(norm)
            simp_word_after, norm_word_after = [], []

            for word in simp_tok:
                if word.isalpha():
                    simp_word_after.append(word)
            for word in norm_tok:
                if word.isalpha():
                    norm_word_after.append(word)
            if len(simp_word_after) > 40 or len(norm_word_after) > 40:
                # print(len(simp_word_after))
                continue

            cache.append(original_file.loc[i, :])


def bleu_filter(from_path, to_path):
    col_names = ['doc_id', 'simp_id', 'norm_id', 'simp_sent', 'norm_sent']
    table = pd.read_csv(from_path, header=None, names=col_names)
    new_table = pd.DataFrame(columns=col_names)

    filtered = []
    # print(new_table)

    smoothie = SmoothingFunction().method4
    tokenizer = nltk.SpaceTokenizer()
    for i in range(len(table)):
        simple_sent, norm_sent = table['simp_sent'][i], table['norm_sent'][i]
        score = sentence_bleu([tokenizer.tokenize(simple_sent)], tokenizer.tokenize(norm_sent),
                               [0.25, 0.25, 0.25, 0.25], smoothing_function=smoothie)
        # if score >= 0.9 or score <= 0.1:
        #     count += 1
        if 0.1 <= score <= 0.9:
            # print(pd.concat(new_table, table.iloc[i, :]))
            new_table = new_table.append(table.iloc[i, :], ignore_index=True)
            # print(len(new_table))
    new_table.reset_index(drop=True)
    new_table.to_csv(to_path, index=False, header=False)


    # print(count / len(table))
    # print(new_table)
        # print(score)

if __name__ == '__main__':
    remove_outliers("../data/output2.txt", "../data/output_final.txt")

    # bleu_filter("../data/output1.txt", "../data/output2.txt")

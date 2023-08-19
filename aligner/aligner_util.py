import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from cleantext import clean
import re
from unidecode import unidecode

import json
import markdown2
from bs4 import BeautifulSoup

# nltk.download('averaged_perceptron_tagger')
lemmatizer = WordNetLemmatizer()


def markdown_to_text(md_text):
    html_text = markdown2.markdown(md_text, extras=['fenced-code-blocks', 'tables'])
    soup = BeautifulSoup(html_text, 'html.parser')

    pre_tags = soup.findAll('pre')
    for tag in pre_tags:
        tag.string = "<code_large>"

    code_tags = soup.findAll('code')
    for tag in code_tags:
        tag.string = "<code_small>"

    table_tags = soup.findAll('table')
    for tag in table_tags:
        tag.string = "<table>"

    url_tags = soup.findAll('a')
    for tag in url_tags:
        url_text = tag.text.strip()
        tag.replace_with(f'{url_text} <url>')

    for tag in soup(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        tag.extract()

    return soup.get_text()

    #####################
    # word_tokens = nltk.word_tokenize(text.lower())  # tokenize can convert " to ``
    # cleaned_text = []
    #
    # for word in word_tokens:
    #     if word.isalpha():
    #         cleaned_text.append(lemmatizer.lemmatize(word))
    #     else:
    #         cleaned_text.append(word)
    #
    # # return recover_text(cleaned_text)
    # return " ".join(cleaned_text)


def clean_md_text(sentences):

    sentences = remove_symbols(sentences)
    # sentences = mask_code(sentences)
    # sentences = mask_comment(sentences)

    return sentences


def remove_symbols(sentences):
    cleaned_sentences = []

    for sentence in nltk.sent_tokenize(sentences):

        sentence = re.sub("#", "", sentence)
        sentence = re.sub("\*", "", sentence)
        sentence = re.sub("'''.+'''", "<codeblock>", sentence)
        sentences = re.sub("```.*```", "<codeblock>", sentences)
        cleaned_sentences.append(sentence)

    return ' '.join(cleaned_sentences)


def mask_url(sentences):
    return sentences


def extract_code_block(sentences):
    return sentences


def clean_sentences(raw_sentences):
    cleaned_sentences = []
    for raw_sentence in raw_sentences:
        if len(nltk.word_tokenize(raw_sentence)) > 5:
            cleaned_sentences.append(raw_sentence)

    return cleaned_sentences


def recover_text(word_list):
    recovered_text = ""
    recovered_list = []
    left_parenthesis = set(['<', '[', '{', '('])
    for i in range(len(word_list)):
        if word_list[i].isalpha() or bool(re.search('[a-zA-Z]]', word_list[i])):
            recovered_list.append(word_list[i])
        elif word_list[i] in left_parenthesis:
            word_list[i+1] = word_list[i] + word_list[i+1]

    return recovered_text


if __name__ == '__main__':
    pass

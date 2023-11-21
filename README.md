## Evaluating Transfer Learning for Simplifying GtiHub READMEs

This is the code for paper "Evaluating Transfer Learning for Simplifying GitHub READMEs".

Please kindly find our preprint paper on [Arxiv](https://arxiv.org/pdf/2308.09940.pdf)


### Introduction
In this paper, we harvested a new software-related simplification dataset. A transformer
model is trained both Wikipedia to Simple Wikipedia dataset as well as our newly proposed dataset.
<br>We experimented with transfer learning, which generates better results.

### Package requirements
In order to run the code, please install packages include:
* PyGithub
* nltk
* pytorch
* numpy
* scipy
* BeautifulSoup
* pytorch-transformers
* pytorch-beam-search

### Folders Walkthrough

#### Github_API
This folder contains code for harvesting data from GitHub.

#### Aligner
`aligner/` contains all steps for preprocessing collected data and perform alignment task.
<br> The BERT checkpoint for doing the alignment task was from Jiang et al. (Neural CRF Model for Sentence Alignment in Text Simplification). We only make some modifications on it to fit for our case. The BERT checkpoint can be accessed throught this [link](https://mega.nz/file/rMJzVRxT#4Hz4mObSrQHI58uajPI52xOBE2YKMn9ZHZ8Z_tci5SA).
<br> E.g. use the following command to align sentences.
```shell
python main.py --ipath=../data/db_eliminated_duplicate.txt --bert=../BERT_wiki --opath="../data/output.txt"
```
#### Simplification
`simplification/` contains training, evaluation and generation steps.
<br>To train the model, use command
```shell
python3 train.py --config=training_config.json --model=model_config.json --save_path=to_path --data_source=wiki
```
You can specify the model configuration in the `model_config.json` file. Hyperparameters are adjustable in 
`train_config.json`. Two data sources are available for training, namely wiki and software simplification corpus.

<br>To generate simplified sentences using a model checkpoint, use command
```shell
python3 generate.py --model=model_checkpoint --path=src_sentence_file --beam=5 --to_path=write_path
```
<br>After generating the simplified sentence, you could use BLEU score to evaluate the model performance, use
command

```shell
python3 evaluate.py --candidate=generated_sentences_file --reference=reference_sentences_file
```

### Model Checkpoints 

To visit the model checkpoints for our survey, go to this [link](https://mega.nz/file/zFIhSYKJ#Xs12CzP8kY3i-RmNVMmcAIjiqLMVRVcUuBO0KlQd1ZA).

### Survey
BLEU score is not ideal for evaluating the simplification system. In this research, we performed a survey. Please find the survey annotation and scores in the corresponding folder.

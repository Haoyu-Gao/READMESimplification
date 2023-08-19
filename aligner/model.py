import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
from pytorch_transformers import BertConfig, BertForSequenceClassification, BertTokenizer

my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Aligner(nn.Module):
    """
    For now the aligner is barely a BERT, but we can add more according to the conditional random field
    """

    def __init__(self, bert_for_seq_model, tokenizer, training=False):
        super(Aligner, self).__init__()
        bert_for_seq_model.to(my_device)
        self.bert_classifier = bert_for_seq_model
        self.tokenizer = tokenizer
        self.training = training

    def forward(self, sents1, sents2):
        """
        please do not call this when doing inference
        """
        sent_A_list = []
        sent_B_list = []
        for i in range(len(sents1)):
            for j in range(len(sents2)):
                sent_A_list.append(sents1[i])
                sent_B_list.append(sents2[j])

        tensor_mat = get_tensor_from_sent_pair(sent_A_list, sent_B_list, self.bert_classifier, self.tokenizer)

        n = int(tensor_mat.size(0) / len(sents1))
        for i in range(len(sents1)):
            for j in range(n):
                print(torch.argmax(tensor_mat[i * n + j]))

        return

    def get_pairs_with_bert(self, sents1, sents2, window_size):
        pair_indices = []
        sent_A_list = []
        sent_B_list = []
        candidate_pairs = []
        for i in range(len(sents1)):
            for j in range(len(sents2)):
                if i - window_size <= j <= i + window_size:
                    sent_A_list.append(sents1[i])
                    sent_B_list.append(sents2[j])
                    candidate_pairs.append((i, j))

        tensor_mat = get_tensor_from_sent_pair(sent_A_list, sent_B_list, self.bert_classifier, self.tokenizer)
        if tensor_mat is not None:

            for i in range(tensor_mat.size(0)):
                if torch.argmax(tensor_mat[i]).item() == 0:
                    pair_indices.append(candidate_pairs[i])

        return pair_indices


class InferenceDataset(Dataset):
    """
    Dataset for inferencing alignment sentence pairs.
    Each corresponding parallel corpus should construct its own InferenceDataset separately.
    """

    def __init__(self, normal_sentences, simple_sentences, tokenizer):
        super(InferenceDataset, self).__init__()
        self.normal_sentences = normal_sentences
        self.simple_sentences = simple_sentences
        self.sent_pairs = self._pair_enumeration()
        self.tokenizer = tokenizer

    def _pair_enumeration(self):
        sent_pairs = []
        for simple_sent in self.simple_sentences:
            for normal_sent in self.normal_sentences:
                sent_pairs.append((simple_sent, normal_sent))
        return sent_pairs

    def __len__(self):
        return len(self.sent_pairs)

    def __getitem__(self, idx):
        simple_sent, normal_sent = self.sent_pairs[idx]

        encoded_instance = self.tokenizer.encode(simple_sent, normal_sent, add_special_tokens=True)
        encoded_instance = torch.tensor(encoded_instance)
        return encoded_instance


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask  # differentiate which part is input, which part is padding
        self.segment_ids = segment_ids  # differentiate different sentences
        self.label_id = label_id


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_tensor_from_sent_pair(sentA, sentB, model, tokenizer, mode='train'):

    model.eval()
    fake_example = []
    for i in range(len(sentA)):
        fake_example.append(InputExample(guid=i, text_a=sentA[i], text_b=sentB[i], label='good'))

    fake_example_features = convert_examples_to_features(fake_example, ["good", 'bad'], 128, tokenizer,
                                                         'classification',
                                                         cls_token_at_end=bool('bert' in ['xlnet']),
                                                         # xlnet has a cls token at the end
                                                         cls_token=tokenizer.cls_token,
                                                         cls_token_segment_id=2 if 'bert' in ['xlnet'] else 0,
                                                         sep_token=tokenizer.sep_token,
                                                         sep_token_extra=bool('bert' in ['roberta']),
                                                         # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                         pad_on_left=bool('bert' in ['xlnet']),
                                                         # pad on the left for xlnet
                                                         pad_token=
                                                         tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                         pad_token_segment_id=4 if 'bert' in ['xlnet'] else 0,
                                                         )

    all_input_ids = torch.tensor([f.input_ids for f in fake_example_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in fake_example_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in fake_example_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in fake_example_features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=256)
    for batch in eval_dataloader:
        batch = tuple(t.to(my_device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}
            outputs = model(input_ids=inputs["input_ids"], \
                            attention_mask=inputs["attention_mask"], \
                            token_type_ids=inputs["token_type_ids"], \
                            labels=None, \
                            )

            output_tensor.append(outputs[0])
    try:
        output_tensor = torch.cat(output_tensor)
    except:
        return None

    return output_tensor


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('../BERT_wiki/', do_lower_case=True)
    sequence_classifier = BertForSequenceClassification.from_pretrained('../BERT_wiki/', output_hidden_states=False)

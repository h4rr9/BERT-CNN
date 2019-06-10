import re
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm
from utils import tokenization

_VOCAB_FILE = './data/uncased_L-24_H-1024_A-16/vocab.txt'
# _VOCAB_FILE = '/scratch/scratch1/harig/data/uncased_L-24_H-1024_A-16/vocab.txt'
_DO_LOWER_CASE = True


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b=None):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


def preprocess_label(label, onehot=False):

    if onehot:
        label = LabelEncoder().fit_transform(label)
        return OneHotEncoder().fit_transform(label.reshape(-1, 1)).toarray()

    return LabelEncoder().fit_transform(label)


def process_label(label=None, classification=True):

    if not label is None:
        if classification:
            print('##### {0}'.format(len(np.unique(label))))
            if len(np.unique(label)) > 2:
                label = preprocess_label(label, onehot=True)
            else:
                label = preprocess_label(label, onehot=False)
    return label


def create_tokenizer():
    return tokenization.FullTokenizer(vocab_file=_VOCAB_FILE, do_lower_case=_DO_LOWER_CASE)


def convert_text_to_example(texts):
    examples = []

    for uid, text in enumerate(texts):
        line = tokenization.convert_to_unicode(text)

        if not line:
            break

        line = line.strip()
        text_a = None
        text_b = None
        m = re.match(r"^(.*) \|\|\| (.*)$", line)

        if m is None:
            text_a = line
        else:
            text_a = m.group(1)
            text_b = m.group(2)

        examples.append(InputExample(
            unique_id=uid, text_a=text_a, text_b=text_b))

    return examples


def convert_single_example(tokenizer, example, seq_length):

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        _truncated_seq_pair(tokens_a, tokens_b, seq_length - 3)
    else:
        if len(tokens_a) > seq_length - 2:
            tokens_a = tokens_a[0:(seq_length - 2)]

    tokens = []
    input_type_ids = []
    tokens.append('[CLS]')
    input_type_ids.append(0)

    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(0)
    tokens.append('[SEP]')
    input_type_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            input_type_ids.append(1)
        tokens.append('[SEP]')
        input_type_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    return input_ids, input_mask, input_type_ids


def convert_examples_to_features(tokenizer, examples, max_seq_length=256):

    input_ids, input_masks, input_type_ids = [], [], []

    for example in tqdm(examples, desc="Converting examples to features"):
        input_id, input_mask, input_type_id = convert_single_example(
            tokenizer, example, seq_length=max_seq_length)

        input_ids.append(input_id)
        input_masks.append(input_mask)
        input_type_ids.append(input_type_id)

    return np.array([np.array(input_ids), np.array(input_type_ids), np.array(input_masks)])


def _truncated_seq_pair(tokens_a, tokens_b, max_length):

    while True:
        total_length = len(tokens_a) + len(tokens_b)

        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

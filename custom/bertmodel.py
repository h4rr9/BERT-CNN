import keras_bert
import os
from keras.models import Model

# _BERT_PATH = '/scratch/scratch1/harig/data'
_BERT_PATH = './data'
_BERT_LARGE = 'uncased_L-24_H-1024_A-16'
_BERT_BASE = 'uncased_L-12_H-768_A-12'

_BERT_SPEC = os.path.join(_BERT_PATH, _BERT_BASE)


def BERTModel(seq_len=128, trainable=False):

    config_path = os.path.join(_BERT_SPEC, 'bert_config.json')
    checkpoint_path = os.path.join(_BERT_SPEC, 'bert_model.ckpt')

    model = keras_bert.load_trained_model_from_checkpoint(
        config_file=config_path, checkpoint_file=checkpoint_path, training=trainable, seq_len=seq_len)

    return model

from base.base_model import BaseModel
from tensorflow.keras.models import Model
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from utils.metrics import matt_corr
from utils import custom_layers


class ConvCoLAModel(BaseModel):
    def __init__(self, config):
        super(ConvCoLAModel, self).__init__(config)
        self.build_model(64)

    def build_model(self, max_seq_length):

        _in = layers.Input(shape=(3, max_seq_length,), name='BERT_input')

        in_id = layers.Lambda(lambda x: x[:, 0, :], output_shape=(
            None, max_seq_length,), name='input_ids')(_in)

        in_mask = layers.Lambda(lambda x: x[:, 1, :], output_shape=(
            None, max_seq_length,), name='input_masks')(_in)

        in_type_id = layers.Lambda(lambda x: x[:, 2, :], output_shape=(
            None, max_seq_length,), name='input_type_ids')(_in)

        bert_inputs = [in_id, in_mask, in_type_id]

        bert_output = custom_layers.BertLayer()(bert_inputs)

        conv1 = layers.Conv1D(filters=100, kernel_size=3,
                              padding='same', name='conv1')(bert_output)
        pool1 = layers.GlobalMaxPool1D()(conv1)

        conv2 = layers.Conv1D(filters=100, kernel_size=4,
                              padding='same', name='conv2')(bert_output)
        pool2 = layers.GlobalMaxPool1D()(conv2)

        conv3 = layers.Conv1D(filters=100, kernel_size=5,
                              padding='same', name='conv3')(bert_output)
        pool3 = layers.GlobalMaxPool1D()(conv3)

        concat0 = layers.Concatenate()([pool1, pool2, pool3])

        preds = layers.Dense(1, activation='sigmoid')(concat0)

        self.model = models.Model(inputs=_in, outputs=preds)

        self.model.compile(loss='binary_crossentropy',
                           optimizer=self.config.model.optimizer, metrics=['accuracy', matt_corr])



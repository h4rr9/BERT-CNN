from keras import models
from keras import layers
import custom
import os


class BaseModel(object):
    def __init__(self, config):
        self.config = config

    def build_bert_layer(self, max_seq_length):
        self._bert = custom.BERTModel(
            seq_len=max_seq_length, trainable=False)

        extract_layer = custom.RemoveMask()(
            self._bert.layers[-1].output)

        return extract_layer

    def build_base_model(self, max_seq_length):

        dropout_rate = self.config.model.dropout_rate
        activation = self.config.model.activation

        embedding = self.build_bert_layer(max_seq_length=max_seq_length)

        block1 = custom.GatedBlock(
            inputs=embedding, filters=32, name='block1')

        pool1 = layers.MaxPool1D(pool_size=2)(block1)
        x = layers.BatchNormalization(axis=-1)(pool1)
        dropout1 = layers.Dropout(rate=dropout_rate)(x)

        block2 = custom.GatedBlock(
            inputs=dropout1, filters=64, name='block2')

        pool2 = layers.MaxPool1D(pool_size=2)(block2)
        x = layers.BatchNormalization(axis=-1)(pool2)
        dropout2 = layers.Dropout(rate=dropout_rate)(x)

        block3 = custom.GatedBlock(
            inputs=dropout2, filters=128, name='block3')

        pool3 = layers.MaxPool1D(pool_size=2)(block3)
        x = layers.BatchNormalization(axis=-1)(pool3)
        dropout3 = layers.Dropout(rate=dropout_rate)(x)

        flatten = layers.GlobalAveragePooling1D()(dropout3)

        x = layers.Dense(units=128, activation=activation)(flatten)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Dropout(rate=dropout_rate)(x)

        x = layers.Dense(units=64, activation=activation)(x)
        x = layers.BatchNormalization(axis=-1)(x)
        self.base_out = layers.Dropout(rate=dropout_rate)(x)

    def build_model(self):
        raise NotImplementedError

    def save_weights(self, checkpoint_path):
        if self.model is None:
            raise Exception("ModelNotBuiltException")

        print("Saving model weights.")
        self.model.save_weights(os.path.join(checkpoint_path, "weights.h5"))

    def save_config(self, checkpoint_path):
        if self.model is None:
            raise Exeption("ModelNotBuiltException")

        print("Saving model config.")
        json_config = self.model.to_json()

        with open(os.path.join(checkpoint_path, "config.json"), "w") as jsonfile:
            jsonfile.write(json_config)

    def save_model(self, checkpoint_path):
        if self.model is None:
            raise Exeption("ModelNotBuiltException")

        print("Saving model.")

        models.save_model(self.model, os.path.join(
            checkpoint_path, "model.h5"))

    def load_weights(self, checkpoint_path):
        if self.model is None:
            raise Exeption("ModelNotBuiltException")

        print("Loading model weights.")

        self.model.load_weights(os.path.join(checkpoint_path, "weights.h5"))

    def load_config(self, checkpoint_path):

        print("Loading model config.")

        with open(os.path.join(checkpoint_path, "config.json"), "r") as json_file:
            self.model = models.model_from_json(
                json_file)

    def load_model(self, checkpoint_path):

        print("Loading model.")

        self.model.load_model(os.path.join(checkpoint_path, "model.h5"))

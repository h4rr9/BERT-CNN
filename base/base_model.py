import tensorflow as tf
import custom_layers
import os

import custom_layers


class BaseModel(object):
    def __init__(self, config):
        self.config = config

    def build_bert_layer(self, max_seq_length):

        self.base_in = tf.keras.layers.Input(
            shape=(3, max_seq_length,), name='BERT_input')

        in_id = tf.keras.layers.Lambda(lambda x: x[:, 0, :], output_shape=(
            None, max_seq_length,), name='input_ids')(self.base_in)

        in_mask = tf.keras.layers.Lambda(lambda x: x[:, 1, :], output_shape=(
            None, max_seq_length,), name='input_masks')(self.base_in)

        in_type_id = tf.keras.layers.Lambda(lambda x: x[:, 2, :], output_shape=(
            None, max_seq_length,), name='input_type_ids')(self.base_in)

        bert_inputs = [in_id, in_mask, in_type_id]

        bert_output = custom_layers.BertLayer()(bert_inputs)

        return bert_output

    def build_base_model(self, max_seq_length):

        embedding = self.build_bert_layer(max_seq_length=max_seq_length)

        block1 = custom_layers.GatedBlock(
            inputs=embedding, filters=32, name='block1')

        pool1 = tf.keras.layers.MaxPool1D(pool_size=2)(block1)
        x = tf.keras.layers.BatchNormalization(axis=-1)(pool1)
        dropout1 = tf.keras.layers.Dropout(rate=0.5)(x)

        block2 = custom_layers.GatedBlock(
            inputs=dropout1, filters=64, name='block2')

        pool2 = tf.keras.layers.MaxPool1D(pool_size=2)(block2)
        x = tf.keras.layers.BatchNormalization(axis=-1)(pool2)
        dropout2 = tf.keras.layers.Dropout(rate=0.5)(x)

        block3 = custom_layers.GatedBlock(
            inputs=dropout2, filters=128, name='block3')

        pool3 = tf.keras.layers.MaxPool1D(pool_size=2)(block3)
        x = tf.keras.layers.BatchNormalization(axis=-1)(pool3)
        dropout3 = tf.keras.layers.Dropout(rate=0.5)(x)

        flatten = tf.keras.layers.GlobalAveragePooling1D()(dropout3)

        x = tf.keras.layers.Dense(units=128, activation='elu')(flatten)
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
        x = tf.keras.layers.Dropout(rate=0.5)(x)

        x = tf.keras.layers.Dense(units=64, activation='elu')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)

        self.base_out = tf.keras.layers.Dropout(rate=0.5)(x)

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

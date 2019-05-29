from tensorflow.keras import models
from tensorflow.keras import layers
from utils import custom_layers
import os


class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.build_base_model()

    def build_base_model(self, max_seq_length):

        self.base_in = layers.Input(
            shape=(3, max_seq_length,), name='BERT_input')

        in_id = layers.Lambda(lambda x: x[:, 0, :], output_shape=(
            None, max_seq_length,), name='input_ids')(self.base_in)

        in_mask = layers.Lambda(lambda x: x[:, 1, :], output_shape=(
            None, max_seq_length,), name='input_masks')(self.base_in)

        in_type_id = layers.Lambda(lambda x: x[:, 2, :], output_shape=(
            None, max_seq_length,), name='input_type_ids')(self.base_in)

        bert_inputs = [in_id, in_mask, in_type_id]

        bert_output = custom_layers.BertLayer()(bert_inputs)

        # model begins here

        convx = layers.Conv1D(filters=100, kernel_size=3,
                              padding='same', name='conv1')(bert_output)
        poolx = layers.GlobalMaxPool1D()(convx)

        convy = layers.Conv1D(filters=100, kernel_size=4,
                              padding='same', name='conv2')(bert_output)
        pooly = layers.GlobalMaxPool1D()(convy)

        convz = layers.Conv1D(filters=100, kernel_size=5,
                              padding='same', name='conv3')(bert_output)
        poolz = layers.GlobalMaxPool1D()(convz)

        self.base_out = layers.Concatenate()([poolx, pooly, poolz])

        # final layer in derived mdoel class

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

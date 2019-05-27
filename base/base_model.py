from tensorflow.keras import models
import os


class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.model = None

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

    def build_model(self):
        raise NotImplementedError

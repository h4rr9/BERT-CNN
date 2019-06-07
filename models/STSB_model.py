from base.base_model import BaseModel
import tensorflow as tf
from utils.metrics import pear_corr, spear_corr


class STSBModel(BaseModel):
    def __init__(self, config):
        super(STSBModel, self).__init__(config)
        self.build_base_model(max_seq_length=128)
        self.build_model()

    def build_model(self):

        preds = tf.keras.layers.Dense(
            units=1, activation='linear')(self.base_out)

        self.model = tf.keras.models.Model(inputs=self.base_in, outputs=preds)

        self.model.compile(loss='mse',
                           optimizer=self.config.model.optimizer, metrics=[pear_corr])

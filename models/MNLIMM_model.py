from base.base_model import BaseModel
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers


class MNLIMMModel(BaseModel):
    def __init__(self, config):
        super(MNLIMMModel, self).__init__(config)
        self.build_base_model(max_seq_length=128)
        self.build_model()

    def build_model(self):

        preds = layers.Dense(units=3, activation='softmax')(self.base_out)

        self.model = models.Model(inputs=self.base_in, outputs=preds)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.config.model.optimizer, metrics=['accuracy'])

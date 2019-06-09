
from base.base_model import BaseModel
from keras import models
from keras import layers


class RTEModel(BaseModel):
    def __init__(self, config):
        super(RTEModel, self).__init__(config)
        self.build_base_model(max_seq_length=128)
        self.build_model()

    def build_model(self):

        preds = layers.Dense(units=1, activation='sigmoid')(self.base_out)

        self.model = models.Model(inputs=self._bert.input, outputs=preds)

        self.model.compile(loss='binary_crossentropy',
                           optimizer=self.config.model.optimizer, metrics=['accuracy'])

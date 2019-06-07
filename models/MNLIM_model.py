from base.base_model import BaseModel
import tensorflow as tf


class MNLIMModel(BaseModel):
    def __init__(self, config):
        super(MNLIMModel, self).__init__(config)
        self.build_base_model(max_seq_length=128)
        self.build_model()

    def build_model(self):

        preds = tf.keras.layers.Dense(units=3, activation='softmax')(self.base_out)

        self.model = tf.keras.models.Model(inputs=self.base_in, outputs=preds)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.config.model.optimizer, metrics=['accuracy'])

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import backend as K


_BERT_SPEC = './data/bert_module/'


class BertLayer(tf.layers.Layer):

    def __init__(self, n_fine_tune_layers=None, **kwargs):

        super(BertLayer, self).__init__(**kwargs)
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True if n_fine_tune_layers else False
        self.output_size = 768

    def build(self, input_shape):
        self.bert = hub.Module(
            spec=_BERT_SPEC,
            trainable=self.trainable
        )

        if self.trainable:
            trainable_vars = self.bert.variables

            # Remove unused layers
            trainable_vars = [
                var for var in trainable_vars if not "/cls/" in var.name]

            # Select how many layers to fine tune
            trainable_vars = trainable_vars[-self.n_fine_tune_layers:]

            # Add to trainable weights
            for var in trainable_vars:
                self._trainable_weights.append(var)

            for var in self.bert.variables:
                if var not in self._trainable_weights:
                    self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
            "sequence_output"
        ]
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

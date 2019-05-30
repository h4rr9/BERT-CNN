import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

class GatedConv1D(models.Model):
  def __init__(self, kernel_size, filters, residual=False, name=None):
    super(GatedConv1D, self).__init__(name=name)
    self.residual = residual
    self.zeropadding = layers.ZeroPadding1D(padding=(kernel_size - 1, 0))
    self.conva = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding="valid", name="conva")
    self.convb = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=1, activation="sigmoid", padding="valid", name="convb")
    self.multiply = layers.Multiply()
    self.convc = layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding="valid", name="conv1x1")

  def call(self, input_tensor):
    x_input = self.zeropadding(input_tensor)
    c1 = self.conva(x_input)
    c2 = self.convb(x_input)

    x_output = self.multiply([c1, c2])

    if self.residual:
      if input_tensor.shape[-1] == x_output.shape[-1]:
        x_output += input_tensor
      else:
        residual = self.convc(input_tensor)
        x_output += residual

    return tf.nn.relu(x_output)

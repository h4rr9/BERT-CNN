import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models


class GatedConv1D(models.Model):
    def __init__(self, kernel_size, filters, residual=False, name=None):
        super(GatedConv1D, self).__init__(name=name)
        self.residual = residual
        self.zeropadding = layers.ZeroPadding1D(padding=(kernel_size - 1, 0))
        self.conva = layers.Conv1D(
            filters=filters, kernel_size=kernel_size, strides=1, padding="valid", name="conva")
        self.convb = layers.Conv1D(filters=filters, kernel_size=kernel_size,
                                   strides=1, activation="sigmoid", padding="valid", name="convb")
        self.multiply = layers.Multiply()
        self.convc = layers.Conv1D(
            filters=filters, kernel_size=1, strides=1, padding="valid", name="conv1x1")

    def call(self, input_tensor):
        padded_input = self.zeropadding(input_tensor)
        c1 = self.conva(padded_input)
        c2 = self.convb(padded_input)

        x_output = self.multiply([c1, c2])

        if self.residual:
            if input_tensor.shape[-1] == x_output.shape[-1]:
                x_output += input_tensor
            else:
                residual = self.convc(input_tensor)
                x_output += residual

        return tf.nn.relu(x_output)


class GatedConv1D_bottleneck(models.Model):
    def __init__(self, kernel_size, filters, bottleneck_size=None, residual=False, name=None):
        super(GatedConv1D_bottleneck, self).__init__(name=name)

        if bottleneck_size is None:
            bottleneck_size = filters // 2

        self.conv1x1a = layers.Conv1D(
            filters=bottleneck_size, kernel_size=1, strides=1, padding="valid", name="conva1x1a")

        self.zeropadding = layers.ZeroPadding1D(padding=(kernel_size - 1, 0))
        self.conva = layers.Conv1D(
            filters=filters, kernel_size=bottleneck_size, strides=1, padding="valid", name="conva")
        self.convb = layers.Conv1D(filters=bottleneck_size, kernel_size=kernel_size,
                                   strides=1, activation="sigmoid", padding="valid", name="convb")
        self.multiply = layers.Multiply()

        self.conv1x1b = layers.Conv1D(
            filters=filters, kernel_size=1, strides=1, padding="valid", name="conva1x1b")

        self.conv1x1c = layers.Conv1D(
            filters=filters, kernel_size=1, strides=1, padding="valid", name="conva1x1c")

    def call(self, input_tensor):

        x_input = self.conv1x1a(input_tensor)

        padded_input = self.zeropadding(x_input)
        c1 = self.conva(padded_input)
        c2 = self.convb(padded_input)

        product = self.multiply([c1, c2])

        x_output = self.conv1x1b(product)

        if input_tensor.shape[-1] == x_output.shape[-1]:
            x_output += input_tensor
        else:
            residual = self.conv1x1c(input_tensor)
            x_output += residual

        return tf.nn.relu(x_output)

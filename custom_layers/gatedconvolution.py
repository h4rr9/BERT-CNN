import tensorflow as tf


def GatedConv1D(inputs, kernel_size, filters, name='gatedconv'):
    with tf.keras.backend.name_scope(name):

        padded_input = layers.ZeroPadding1D(
            padding=(kernel_size - 1, 0))(inputs)

        conv1 = layers.Conv1D(filters=filters, kernel_size=kernel_size,
                              strides=1, padding="valid")(padded_input)
        conv2 = layers.Conv1D(filters=filters, kernel_size=kernel_size,
                              strides=1, activation="sigmoid", padding="valid")(padded_input)

        output = layers.Multiply()([conv1, conv2])

    return output


def GatedBlock(inputs, filters, name='gatedblock'):
    with tf.keras.backend.name_scope(name):

        conv1 = GatedConv1D(inputs=inputs, kernel_size=3,
                            filters=filters, name='gatedconv3x3')

        conv2 = GatedConv1D(inputs=inputs, kernel_size=5,
                            filters=filters, name='gatedconv5x5')

        output = layers.Concatenate()([conv1, conv2])

    return output

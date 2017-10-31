import tensorflow as tf
from tensorflow.contrib import rnn


class CNN(object):
    def __init__(self, kw, feature_maps, kernels, in_channels, scope_name, stddev=0.02):
        self.Ws = []
        assert len(kernels) == len(feature_maps)
        for idx in range(len(kernels)):
            with tf.variable_scope(scope_name):
                W = tf.get_variable(
                    'kernel_W_%d' % idx,
                    shape=[kernels[idx], kw, in_channels, feature_maps[idx]],
                    initializer=tf.truncated_normal_initializer(stddev=stddev)
                )
            self.Ws.append(W)

    def __call__(self, _input):
        layers = []
        for W in self.Ws:
            conv = tf.nn.conv2d(
                _input,
                filter=W,
                strides=[1, 1, 1, 1],
                padding='VALID',
            )
            conv = tf.tanh(conv)
            after_pool = tf.nn.max_pool(
                conv,
                [1, conv.shape[1], 1, 1],
                [1, 1, 1, 1],
                padding='VALID'
            )
            layers.append(tf.reshape(after_pool, [-1, int(after_pool.shape[-1])]))
        return tf.concat(layers, 1)


class LSTM(object):
    def __init__(self, dim, fw_forget_bias, bw_forget_bias, scope_name):
        with tf.variable_scope(scope_name):
            self.fw_LSTM = rnn.BasicLSTMCell(
                num_units=dim,
                forget_bias=fw_forget_bias
            )
            self.bw_LSTM = rnn.BasicLSTMCell(
                num_units=dim,
                forget_bias=bw_forget_bias
            )

    def __call__(self, input_, layer_size):
        for _ in range(layer_size):
            (fw_output, bw_output),_ = tf.nn.bidirectional_dynamic_rnn(
                self.fw_LSTM,
                self.bw_LSTM,
                input_,
                time_major=False,
                dtype=tf.float32
            )
            input_ = tf.concat((fw_output, bw_output), axis=2)
        return input_


class highway(object):
    def __init__(self, dim, layer_size, scope_name, gate_f=tf.tanh,
                 fc_f=tf.sigmoid, stddev=0.2):
        with tf.variable_scope(scope_name):
            self.gate_W = tf.get_variable(
                initializer=tf.truncated_normal_initializer(stddev),
                name='gate_W',
                shape=[dim, dim],
                dtype=tf.float32
            )

            self.gate_b = tf.get_variable(
                initializer=tf.truncated_normal_initializer(stddev),
                name='gate_b',
                shape=[dim],
                dtype=tf.float32
            )

        self.gate_f = gate_f
        self.fc_f = fc_f

        self.fc = []

        with tf.variable_scope(scope_name):
            for idx in range(layer_size):
                W = tf.get_variable(
                    initializer=tf.truncated_normal_initializer(stddev),
                    name='fc_W_%d' % idx,
                    shape=[dim, dim],
                    dtype=tf.float32
                )
                b = tf.get_variable(
                    initializer=tf.truncated_normal_initializer(stddev),
                    name='fc_b_%d' % idx,
                    shape=[dim],
                    dtype=tf.float32
                )
                self.fc.append([W, b])

    def __call__(self, input_):

        gate = self.gate_f(input_ @ self.gate_W + self.gate_b)

        output_ = input_

        for W, b in self.fc:
            output_ = self.fc_f(output_ @ W + b)

        return gate * output_ + input_




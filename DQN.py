import tensorflow as tf
import numpy as np


class DQN(object):
    def __init__(self, sess, window_size, input_shape, num_actions, num_atoms, name='dqn',double=1, duel=0, dist=0, noisy=0, trainable=True):
        self.inputs = tf.placeholder('float32', [None,input_shape[0], input_shape[1], window_size ])
        self.sess = sess
        self.copy_op = None
        self.name = name
        self.var = {}
        self.double = double
        self.duel = duel
        self.dist = dist
        self.noisy = noisy
        self.weights_initializer = tf.contrib.layers.xavier_initializer()
        self.biases_initializer = tf.constant_initializer(0.1)
        self.num_atoms = num_atoms
        self.num_actions = num_actions

        with tf.variable_scope(name):
            # Layer 1 (Convolution)
            self.l1, self.var['l1_w'], self.var['l1_b'] = conv2d(self.inputs, 32, "l1_conv", [8, 8], [4, 4])

            # Layer 2 (Convolution)
            self.l2, self.var['l2_w'], self.var['l2_b'] = conv2d(self.l1, 64, "l2_conv", [4, 4], [2, 2])

            # Layer 3 (Convolution)
            self.l3, self.var['l3_w'], self.var['l3_b'] = conv2d(self.l2, 64, "l3_conv", [3, 3], [1, 1])

            #l3_reshpae = tf.reshape(self.l3, shape=[self.l3.get_shape().as_list()[0], -1], name='flatten')
            dim = self.l3.get_shape().as_list()
            l3_reshape = tf.reshape(self.l3, [-1, dim[1] * dim[2] * dim[3]])

            #l2_concat = tf.concat([l2_reshape, self.aux_inputs], axis=1)

            if self.duel:
                # value function
                if self.noisy:
                    self.l4_value, self.var['l4_value_w'], self.var['l4_value_b'], self.var['l4_value_noise_w'], self.var['l4_value_noise_b'] \
                        = noisy_dense(l3_reshape, 256, "l4_value_fc", trainable=trainable)
                    self.out_value, self.var['out_value_w'], self.var['out_value_b'], self.var['out_value_noise_w'], self.var['out_value_noise_b'] \
                        = noisy_dense(self.l4_value, 1 * num_atoms, "l5_value_fc", activation_fn=None, trainable=trainable)
                    # advantage
                    self.l4_adv, self.var['l4_adv_w'], self.var['l4_adv_b'], self.var['l4_adv_noise_w'], self.var['l4_adv_noise_b'] \
                        = noisy_dense(l3_reshape, 256, "l4_adv_fc", trainable=trainable)
                    self.out_adv, self.var['out_adv_w'], self.var['out_adv_b'], self.var['out_adv_noise_w'], self.var['out_adv_noise_b'] \
                        = noisy_dense(self.l4_adv, self.num_actions * num_atoms, "l5_adv_fc", activation_fn=None, trainable=trainable)
                else:
                    self.l4_value, self.var['l4_value_w'], self.var['l4_value_b'] = dense(l3_reshape, 256, "l4_value_fc", trainable=trainable)
                    self.out_value, self.var['out_value_w'], self.var['out_value_b'] = dense(self.l4_value, 1 * num_atoms,
                                                                                             "l5_value_fc", activation_fn=None, trainable=trainable)
                    # advantage
                    self.l4_adv, self.var['l4_adv_w'], self.var['l4_adv_b'] = dense(l3_reshape, 256, "l4_adv_fc", trainable=trainable)
                    self.out_adv, self.var['out_adv_w'], self.var['out_adv_b'] = dense(self.l4_adv, self.num_actions * num_atoms,
                                                                                       "l5_adv_fc", activation_fn=None, trainable=trainable)

                if self.dist:
                    out_adv_reshape = tf.reshape(self.out_adv, [-1, self.num_actions, num_atoms])
                    out_value_reshape = tf.reshape(self.out_value, [-1, 1, num_atoms])
                    self.outputs = out_value_reshape + (out_adv_reshape - tf.reduce_mean(out_adv_reshape, reduction_indices=1,
                                                                           keep_dims=True))
                else:
                    self.outputs = self.out_value + (self.out_adv - tf.reduce_mean(self.out_adv, reduction_indices=1,
                                                                                   keep_dims=True))
            else:
                if self.noisy:
                    # Layer 3 (FC)
                    self.l4, self.var['l4_w'], self.var['l4_b'], self.var['l4_noise_w'], self.var['l4_noise_b'] \
                        = noisy_dense(l3_reshape, 256, "l4_fc", trainable=trainable)

                    # Layer 4 (FC)
                    self.outputs, self.var['out_w'], self.var['out_b'], self.var['out_noise_w'], self.var['out_noise_b'] \
                        = noisy_dense(self.l4, self.num_actions * num_atoms, "l5_fc", activation_fn=None, trainable=trainable)
                else:
                    # Layer 3 (FC)
                    self.l4, self.var['l4_w'], self.var['l4_b'] = dense(l3_reshape, 256, "l4_fc", trainable=trainable)

                    # Layer 4 (FC)
                    self.outputs, self.var['out_w'], self.var['out_b'] = dense(self.l4, self.num_actions * num_atoms, "l5_fc", activation_fn=None, trainable=trainable)

            self.outputs_idx = tf.placeholder('int32', [None, None], name='outputs_idx')

            if self.dist:
                outputs_quant = tf.reduce_mean(self.outputs, axis=-1)
                self.actions = tf.argmax(outputs_quant, axis=1)
            else:
                self.actions = tf.argmax(self.outputs, axis=1)

            self.max_outputs = tf.reduce_max(self.outputs, reduction_indices=1)  # [batch_size,  nb_atoms]
            self.outputs_with_idx = tf.gather_nd(self.outputs, self.outputs_idx)  # [batch_size,  nb_atoms]


    def run_copy(self):
        if self.copy_op is None:
            raise Exception("run 'create_copy_op first before copy")
        else:
            self.sess.run(self.copy_op)

    def create_copy_op(self, network):
        with tf.variable_scope(self.name):
            copy_ops = []

            for name in self.var.keys():
                copy_op = self.var[name].assign(network.var[name])
                copy_ops.append(copy_op)

            self.copy_op = tf.group(*copy_ops, name='copy_op')

    def calc_actions(self, observation):
        return self.actions.eval({self.inputs: observation}, session=self.sess)

    def calc_outputs(self, observation):
        return self.outputs.eval({self.inputs: observation}, session=self.sess)

    def calc_max_outputs(self, observation):
        return self.max_outputs.eval({self.inputs: observation}, session=self.sess)

    def calc_outputs_with_idx(self, observation, idx):
        return self.outputs_with_idx.eval(
            {self.inputs: observation, self.outputs_idx: idx}, session=self.sess)


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

# wrapper layers
def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="VALID",
           dtype=tf.float32, collections=None,
           activation_fn=tf.nn.relu,
           weights_initializer=tf.contrib.layers.xavier_initializer(),
           biases_initializer=tf.constant_initializer(0.1), trainable=True):

    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        w = tf.get_variable("W", shape=filter_shape, dtype=dtype, initializer=weights_initializer,
                            collections=collections, trainable=trainable)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=biases_initializer,
                            collections=collections, trainable=trainable)
        out = tf.nn.conv2d(x, w, stride_shape, pad) + b
        if activation_fn is not None:
            out = activation_fn(out)
        return out, w, b


def dense(x, size, name, bias=True, activation_fn=tf.nn.relu, trainable=True):
    with tf.variable_scope(name):
        w = tf.get_variable("W", [x.get_shape()[1], size], initializer=normalized_columns_initializer(0.01), trainable=trainable)
        b = None
        out = tf.matmul(x, w)
        if bias:
            b = tf.get_variable("b", [size], initializer=tf.zeros_initializer(), trainable=trainable)
            out = out + b

        if activation_fn is not None:
            out = activation_fn(out)

        return out, w, b


def noisy_dense(x, size, name, bias=True, activation_fn=tf.nn.relu, trainable=True):
    with tf.variable_scope(name):
        input_size = int(x.get_shape()[1])

        noise_factor1 = tf.random_normal([input_size, 1])
        noise_factor2 = tf.random_normal([1, size])
        factored_noise = tf.matmul(noise_factor1, noise_factor2)

        w = tf.get_variable("W", [input_size, size], initializer=tf.random_uniform_initializer(-1/np.sqrt(input_size), 1/np.sqrt(input_size)), trainable=trainable)
        noise_w = tf.get_variable("noise_W", [input_size, size], initializer=tf.constant_initializer(0.5/np.sqrt(input_size)), trainable=trainable)
        out = tf.matmul(x, w + noise_w * factored_noise)

        b = None
        noise_b = None
        if bias:
            b = tf.get_variable("b", [size], initializer=tf.random_uniform_initializer(-1/np.sqrt(input_size), 1/np.sqrt(input_size)), trainable=trainable)
            noise_b = tf.get_variable("noise_b", [size], initializer=tf.constant_initializer(0.5/np.sqrt(input_size)), trainable=trainable)
            out = out + b + noise_b * noise_factor2

        if activation_fn is not None:
            out = activation_fn(out)

        return out, w, b, noise_w, noise_b

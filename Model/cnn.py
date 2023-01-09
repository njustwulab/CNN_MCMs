# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 21:26:37 2019
stride=1 mix with 2
@author: Peng-jz
"""

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class Model(object):
    def __init__(self, learning_rate=1e-4):
        self._learning_rate = learning_rate

    def inference(self, images):  # The input shape is 250*250*1
        with tf.variable_scope('conv1') as scope:
            kernel = self._create_weights([4, 4, 1, 32])
            conv = self._create_conv2d_V3(images, kernel)
            bias = self._create_bias([32])
            preactivation = tf.nn.bias_add(conv, bias)
            conv1 = tf.nn.elu(preactivation, name=scope.name)
            # self._activation_summary(conv1)#83*83*32;S=3

        with tf.variable_scope('conv2') as scope:
            kernel = self._create_weights([5, 5, 32, 64])
            conv = self._create_conv2d_V2(conv1, kernel)
            bias = self._create_bias([64])
            preactivation = tf.nn.bias_add(conv, bias)
            conv2 = tf.nn.elu(preactivation, name=scope.name)
            # self._activation_summary(conv2)#40*40*64;S=2

        with tf.variable_scope('conv3') as scope:
            kernel = self._create_weights([6, 6, 64, 128])
            conv = self._create_conv2d_V2(conv2, kernel)
            bias = self._create_bias([128])
            preactivation = tf.nn.bias_add(conv, bias)
            conv3 = tf.nn.elu(preactivation, name=scope.name)
            # self._activation_summary(conv3)#18*18*128;S=2

        with tf.variable_scope('conv4') as scope:
            kernel = self._create_weights([6, 6, 128, 256])
            conv = self._create_conv2d_V2(conv3, kernel)
            bias = self._create_bias([256])
            preactivation = tf.nn.bias_add(conv, bias)
            conv4 = tf.nn.elu(preactivation, name=scope.name)
            # self._activation_summary(conv4)#7*7*256;S=2

        with tf.variable_scope('conv5') as scope:
            kernel = self._create_weights([3, 3, 256, 512])
            conv = self._create_conv2d_V2(conv4, kernel)
            bias = self._create_bias([512])
            preactivation = tf.nn.bias_add(conv, bias)
            conv5 = tf.nn.elu(preactivation, name=scope.name)
            # self._activation_summary(conv5)#3*3*512;S=2

        with tf.variable_scope('conv6') as scope:
            kernel = self._create_weights([2, 2, 512, 1024])
            conv = self._create_conv2d_V1(conv5, kernel)
            bias = self._create_bias([1024])
            preactivation = tf.nn.bias_add(conv, bias)
            conv6 = tf.nn.elu(preactivation, name=scope.name)
            # self._activation_summary(conv6)#2*2*512;S=1

        with tf.variable_scope('conv7') as scope:
            kernel7 = self._create_weights([1, 1, 1024, 1024])
            conv7 = self._create_conv2d_V1(conv6, kernel7)
            bias7 = self._create_bias([1024])
            preactivation = tf.nn.bias_add(conv7, bias7)
            conv7 = tf.nn.elu(preactivation, name=scope.name)
            # self._activation_summary(conv5)#2*2*1024;S=1

        # deconv
        with tf.variable_scope('deconv1') as scope:
            dekernel = self._create_weights([2, 2, 512, 1024])
            output_shape = tf.stack([tf.shape(images)[0], 3, 3, 512])
            deconv = self._create_deconv1(conv7, dekernel, output_shape)
            debias = self._create_bias([512])
            preactivation = tf.nn.bias_add(deconv, debias)
            deconv = tf.nn.elu(preactivation, name=scope.name)
            concat_layer1 = tf.concat([deconv, conv5], axis=-1)
            # self._activation_summary(deconv1)

        with tf.variable_scope('deconv2') as scope:
            dekernel = self._create_weights([3, 3, 512, 1024])
            output_shape = tf.stack([tf.shape(images)[0], 7, 7, 512])
            deconv = self._create_deconv2(concat_layer1, dekernel, output_shape)
            debias = self._create_bias([512])
            preactivation = tf.nn.bias_add(deconv, debias)
            deconv = tf.nn.elu(preactivation, name=scope.name)
            concat_layer2 = tf.concat([deconv, conv4], axis=-1)
            # self._activation_summary(deconv1)

        with tf.variable_scope('deconv3') as scope:
            dekernel = self._create_weights([6, 6, 256, 768])
            output_shape = tf.stack([tf.shape(images)[0], 18, 18, 256])
            deconv = self._create_deconv2(concat_layer2, dekernel, output_shape)
            debias = self._create_bias([256])
            preactivation = tf.nn.bias_add(deconv, debias)
            deconv = tf.nn.elu(preactivation, name=scope.name)
            concat_layer3 = tf.concat([deconv, conv3], axis=-1)
            # self._activation_summary(deconv2)

        with tf.variable_scope('deconv4') as scope:
            dekernel = self._create_weights([6, 6, 128, 384])
            output_shape = tf.stack([tf.shape(images)[0], 40, 40, 128])
            deconv = self._create_deconv2(concat_layer3, dekernel, output_shape)
            debias = self._create_bias([128])
            preactivation = tf.nn.bias_add(deconv, debias)
            deconv = tf.nn.elu(preactivation, name=scope.name)
            concat_layer4 = tf.concat([deconv, conv2], axis=-1)
            # self._activation_summary(deconv3)

        with tf.variable_scope('deconv5') as scope:
            dekernel = self._create_weights([5, 5, 16, 192])
            output_shape = tf.stack([tf.shape(images)[0], 83, 83, 16])
            deconv = self._create_deconv2(concat_layer4, dekernel, output_shape)
            debias = self._create_bias([16])
            preactivation = tf.nn.bias_add(deconv, debias)
            deconv = tf.nn.elu(preactivation, name=scope.name)
            concat_layer5 = tf.concat([deconv, conv1], axis=-1)
            # self._activation_summary(deconv3)

        with tf.variable_scope('deconv6') as scope:
            dekernel = self._create_weights([4, 4, 1, 48])
            output_shape = tf.stack([tf.shape(images)[0], 250, 250, 1])
            deconv = self._create_deconv3(concat_layer5, dekernel, output_shape)
            debias = self._create_bias([1])
            preactivation = tf.nn.bias_add(deconv, debias)  # 250*250*1
            #            deconv6 = tf.nn.elu(preactivation, name=scope.name)
            #            concat_layer6 = tf.concat([deconv6,conv1], axis=-1)
            # self._activation_summary(deconv5)

            #        with tf.variable_scope('deconv7') as scope:
            #            dekernel7 = self._create_weights([1, 1, 1, 32])
            #            output_shape7 = tf.stack([tf.shape(images)[0],250,250,1])
            #            deconv7 = self._create_deconv1(deconv6, dekernel7, output_shape7)
            #            debias7 = self._create_bias([1])
            #            preactivation2 = tf.nn.bias_add(deconv7, debias7)#250*250*1
            reshape = tf.reshape(preactivation, shape=[-1, 250, 250, 1])

        return reshape

    def train(self, loss, global_step):
        tf.summary.scalar('learning_rate', self._learning_rate)
        train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(loss, global_step=global_step)
        return train_op

    def loss(self, y_pre, ys):
        with tf.variable_scope('loss') as scope:
            cross_entropy = tf.cast(tf.subtract(y_pre, ys), tf.float32)
            cost = tf.reduce_mean(tf.square(cross_entropy), name=scope.name)
            tf.add_to_collection("losses", cost)
            loss = tf.add_n(tf.get_collection("losses"))
            tf.summary.scalar('cost', loss)
        return loss

    def accuracy(self, logits, ys):
        with tf.variable_scope('accuracy') as scope:
            accuracy = tf.reduce_mean(tf.cast(tf.abs(tf.subtract(logits, ys)), tf.float32),
                                      name=scope.name)
            tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def acc(self, logits, ys):
        with tf.variable_scope('accuracy') as scope:
            # acc = tf.cast(tf.abs(tf.subtract(logits, ys)), tf.float32)
            acc = 100 * (1 - tf.reduce_mean(tf.abs(tf.subtract(logits, ys) / ys)))
            tf.summary.scalar('accuracy', acc)
        return acc

    def _create_conv2d_V1(self, x, W):
        return tf.nn.conv2d(input=x,
                            filter=W,
                            strides=[1, 1, 1, 1],
                            padding='VALID')

    def _create_conv2d_V2(self, x, W):
        return tf.nn.conv2d(input=x,
                            filter=W,
                            strides=[1, 2, 2, 1],
                            padding='VALID')

    def _create_conv2d_V3(self, x, W):
        return tf.nn.conv2d(input=x,
                            filter=W,
                            strides=[1, 3, 3, 1],
                            padding='VALID')

    def _create_conv2d_V4(self, x, W):
        return tf.nn.conv2d(input=x,
                            filter=W,
                            strides=[1, 4, 4, 1],
                            padding='VALID')

    def _create_deconv1(self, x, W, output_shape):
        return tf.nn.conv2d_transpose(value=x,
                                      filter=W,
                                      output_shape=output_shape,
                                      strides=[1, 1, 1, 1],
                                      padding='VALID')

    def _create_deconv2(self, x, W, output_shape):
        return tf.nn.conv2d_transpose(value=x,
                                      filter=W,
                                      output_shape=output_shape,
                                      strides=[1, 2, 2, 1],
                                      padding='VALID')

    def _create_deconv3(self, x, W, output_shape):
        return tf.nn.conv2d_transpose(value=x,
                                      filter=W,
                                      output_shape=output_shape,
                                      strides=[1, 3, 3, 1],
                                      padding='VALID')

    def _create_deconv5(self, x, W, output_shape):
        return tf.nn.conv2d_transpose(value=x,
                                      filter=W,
                                      output_shape=output_shape,
                                      strides=[1, 5, 5, 1],
                                      padding='VALID')

    def _create_weights(self, shape):
        # Var = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1, dtype=tf.float32))
        # Var = tf.Variable(tf.random_normal_initializer(seed=None, dtype=tf.float32)(shape))
        Var = tf.Variable(tf.glorot_uniform_initializer(seed=None, dtype=tf.float32)(shape))
        # tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.0001)(Var))
        tf.add_to_collection("losses", 0.0001 * tf.reduce_mean(tf.square(Var)))
        return Var

    def _create_bias(self, shape):
        return tf.Variable(tf.constant(0.1, shape=shape, dtype=tf.float32))

    def _activation_summary(self, x):
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

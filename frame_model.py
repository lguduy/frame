#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Build CNN model

@author: liangyu
"""

import tensorflow as tf


def _activation_summary(activations):
    """Helper to create summaries for activations.

        Creates a summary that provides a histogram of activations.
        Creates a summary that measure the sparsity of activations.

    Parameters:
    -----------
        activation : tensor, activation of layer.

    Returns:
    --------
        no return

    """
    tensor_name = activations.op.name
    # tf.summary
    tf.summary.histogram(tensor_name + '/histogram', activations)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(activations))


def _weight_summary(weights, bias):
    """Helper to create summaries for weights and bias.

        Creates a summary that provides a histogram of weights and bias.

    Parameters:
    -----------
        weights : weights of layers.
        bias : bias of layers.

    Returns:
    --------
        no return
    """
    op_name_weights = weights.op.name
    op_name_bias = bias.op.name
    tf.summary.histogram(op_name_weights + '/histogram', weights)
    tf.summary.histogram(op_name_bias + '/histogram', bias)


def inference(images, n_classes, visualize):
    """Build the model

    Parameters:
    -----------
        images : image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
        n_classes : number of classes, number of model output
        visualize : bool, visualize kernal and activations of conv

    Returns:
    --------
        output tensor with the computed logits, float, [batch_size, n_classes], output of model
    """
    # Conv1 + ReLU, kernal: [3 * 3 * 3, 16], strides: 1
    with tf.variable_scope('Conv1') as scope:
        kernal = tf.get_variable('weights',
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))

        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, kernal, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name='Conv_ReLU')
        _activation_summary(conv1)
        _weight_summary(kernal, biases)

        # visualize kernal and activations of conv1
        if visualize:
            with tf.variable_scope('visualization') as scope:
                # visuzlize kernal
                kernal_min = tf.reduce_min(kernal)
                kernal_max = tf.reduce_max(kernal)
                kernal_0_1 = (kernal - kernal_min) / (kernal_max - kernal_min)
                # transpose [3, 3, 3, 16] to [16, 3, 3, 3]
                kernal_0_1_transpose = tf.transpose(kernal_0_1, [3, 0, 1, 2])
                tf.summary.image(scope.name+'/filters',
                                 kernal_0_1_transpose, max_outputs=16)    # summary.image
                # visuzlize activations of conv1
                layer1_image1 = conv1[0:1, :, :, 0:16]    # one image of a batch
                layer1_image1 = tf.transpose(layer1_image1, [3, 1, 2, 0])
                tf.summary.image(scope.name+"/activations",
                                 layer1_image1, max_outputs=16)

    # Max pool1 + Norm1
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm1')

    # Conv2 + ReLU [3 * 3 * 16, 16]
    with tf.variable_scope('Conv2') as scope:
        kernal = tf.get_variable('weights',
                                  shape=[3, 3, 16, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, kernal, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        # conv2 op name: conv2/Conv+ReLU
        conv2 = tf.nn.relu(pre_activation, name='Conv_ReLU')
        _activation_summary(conv2)
        _weight_summary(kernal, biases)

        # visualize activations of conv2
        if visualize:
            with tf.variable_scope('visualization') as scope:
                # visuzlize activations of conv2
                layer2_image1 = conv2[0:1, :, :, 0:16]    # one image of a batch
                layer2_image1 = tf.transpose(layer2_image1, [3, 1, 2, 0])
                tf.summary.image(scope.name+"/activations",
                                 layer2_image1, max_outputs=16)

    # Norm2  + Max pool2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling2')

    # local3(Fully Connection, FC)
    with tf.variable_scope('local3') as scope:
        # reshape pool to [batch_size * dim]
        dim = int(pool2.get_shape()[1]) * int(pool2.get_shape()[2]) * int(pool2.get_shape()[3])
        reshaped_pool2 = tf.reshape(pool2, [-1, dim])

        weights = tf.get_variable('weights',
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.04))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshaped_pool2, weights) + biases, name='ReLU')
        # summary
        _activation_summary(local3)
        _weight_summary(weights, biases)

    #local4 FC
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128,128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='ReLU')
        # summary
        _activation_summary(local4)
        _weight_summary(weights, biases)

    # output (softmax)
    with tf.variable_scope('Output') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        # not have softmax
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='output')
        # summary
        _activation_summary(softmax_linear)
        _weight_summary(weights, biases)

    return softmax_linear


def losses(logits, labels):
    """Compute loss from logits and labels, loss function

    Parameters:
    -----------
        logits: output tensor of model, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, one hot, [batch_size, n_classes]

    Returns:
    --------
        loss tensor of float type
    """
    with tf.variable_scope('Loss'):
        # log likelihood cost with labels one hot
        # if labels not one hot , use sparse_softmax_cross_entropy_with_logits()
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='cross_entropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
        # tf.summary.scalar
        tf.summary.scalar('loss', loss)

    return loss


def trainning(loss, learning_rate):
    """Training op, the op returned by this function is what must be passed to
        'sess.run()' call to cause the model to train.
        Define a optimizer.

    Parameters:
    -----------
        loss : loss tensor, from losses()
        learning_rate : learning rate

    Returns:
    --------
        train_op :  operation for trainning
    """
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)   # Adam
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        # tf.summary.scalar
        tf.summary.scalar('learning_rate', learning_rate)

    return train_op


def evaluation(logits, labels):
    """Get accuracy.

    Parameters:
    -----------
        logits: output tensor of model, float - [batch_size, num_classes].
        labels: labels tensor, int32 - [batch_size, num_classes], one hot

    Returns:
    --------
        average accuracy of batch size data
    """
    with tf.name_scope('Accuracy'):
        labels_ = tf.argmax(labels, 1)                # transform one hot to labels, [batch_size, ]
        top_one = tf.nn.in_top_k(logits, labels_, 1)  # in_top_k
        correct = tf.cast(top_one, tf.float16)
        accuracy = tf.reduce_mean(correct) * 100
        # tf.summary.scalar
        tf.summary.scalar('accuracy__', accuracy)

    return accuracy

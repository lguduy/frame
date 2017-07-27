#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Build CNN model

@author: liangyu
"""

import tensorflow as tf


def inference(images, batch_size, n_classes, visualize):
    """Build the model
<<<<<<< HEAD

    Parameters:
    -----------
=======
    Parameters:
    -----------------
>>>>>>> c80ea2e7674560a2cf52dd7ca54862f4c7379dd0
        images : image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
        batch_size : batch size
        n_classes : number of classes, number of model output
        visualize : bool, visualize kernal and activations of conv

<<<<<<< HEAD
    Returns:
    --------
=======
    Returns :
    -----------
>>>>>>> c80ea2e7674560a2cf52dd7ca54862f4c7379dd0
        output tensor with the computed logits, float, [batch_size, n_classes], output of model
    """
    # Conv1 + ReLU, kernal: [3 * 3 * 3, 16], strides: 1
    with tf.variable_scope('conv1') as scope_conv1:
        kernal = tf.get_variable('weights',
                                  shape = [3, 3, 3, 16],
                                  dtype = tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
<<<<<<< HEAD

=======
            
>>>>>>> c80ea2e7674560a2cf52dd7ca54862f4c7379dd0
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, kernal, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name= scope_conv1.name)
<<<<<<< HEAD


=======
        
        
>>>>>>> c80ea2e7674560a2cf52dd7ca54862f4c7379dd0
        # visualize kernal and activations of conv1
        if visualize:
            with tf.variable_scope('visualization') as scope_visualize:
                # visuzlize kernal
                kernal_min = tf.reduce_min(kernal)
                kernal_max = tf.reduce_max(kernal)
                kernal_0_1 = (kernal - kernal_min) / (kernal_max - kernal_min)
                # transpose [3, 3, 3, 16] to [16, 3, 3, 3]
                kernal_0_1_transpose = tf.transpose(kernal_0_1, [3, 0, 1, 2])
                tf.summary.image(scope_visualize.name+'filters',
                                 kernal_0_1_transpose, max_outputs=16)    # summary.image
                # visuzlize activations of conv1
                layer1_image1 = conv1[0:1, :, :, 0:16]    # one image of a batch
                layer1_image1 = tf.transpose(layer1_image1, [3, 1, 2, 0])
                tf.summary.image(scope_visualize.name+"activations",
                                 layer1_image1, max_outputs=16)

    # Max pool1 + Norm1
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm1')

    # Conv2 + ReLU [3 * 3 * 16, 16]
    with tf.variable_scope('conv2') as scope_conv2:
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
        conv2 = tf.nn.relu(pre_activation, name=scope_conv2.name)
<<<<<<< HEAD

=======
        
>>>>>>> c80ea2e7674560a2cf52dd7ca54862f4c7379dd0
        # visualize activations of conv2
        if visualize:
            with tf.variable_scope('visualization') as scope_visualize:
                # visuzlize activations of conv2
                layer2_image1 = conv2[0:1, :, :, 0:16]    # one image of a batch
                layer2_image1 = tf.transpose(layer2_image1, [3, 1, 2, 0])
                tf.summary.image(scope_visualize.name+"activations",
                                 layer2_image1, max_outputs=16)

    # Norm2  + Max pool2
    with tf.variable_scope('pooling1_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling1')

    # local3(Fully Connection, FC)
    with tf.variable_scope('local3') as scope:
<<<<<<< HEAD
        reshaped_pool2 = tf.reshape(pool2, shape=[batch_size, -1])   # reshape pool2
=======
        reshaped_pool2 = tf.reshape(pool2, shape=[batch_size, -1])    # reshape pool2
>>>>>>> c80ea2e7674560a2cf52dd7ca54862f4c7379dd0
        dim = reshaped_pool2.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshaped_pool2, weights) + biases, name=scope.name)

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
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

    return softmax_linear


def losses(logits, labels):
    """Compute loss from logits and labels, loss function
<<<<<<< HEAD

    Parameters:
    -----------
=======
    Parameters:
    -----------------
>>>>>>> c80ea2e7674560a2cf52dd7ca54862f4c7379dd0
        logits: output tensor of model, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, one hot, [batch_size, n_classes]

    Returns:
<<<<<<< HEAD
    --------
=======
>>>>>>> c80ea2e7674560a2cf52dd7ca54862f4c7379dd0
        loss tensor of float type
    """
    with tf.variable_scope('loss') as scope:
        # log likelihood cost with labels one hot
<<<<<<< HEAD
        # if labels not one hot , use sparse_softmax_cross_entropy_with_logits()
=======
        # if labels not one hot , use sparse_softmax_cross_entropy_with_logits()  
>>>>>>> c80ea2e7674560a2cf52dd7ca54862f4c7379dd0
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)    # summary

    return loss


def trainning(loss, learning_rate):
<<<<<<< HEAD
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
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)   # Adam
=======
    """Training ops, the Op returned by this function is what must be passed to
        'sess.run()' call to cause the model to train
        define a optimizer

    Parameters:
    ----------------
        loss : loss tensor, from losses()
        learning_rate : learning rate
    Returns:
    ------------
        train_op :  operation for trainning
    """
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)    # Adam
>>>>>>> c80ea2e7674560a2cf52dd7ca54862f4c7379dd0
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def evaluation(logits, labels):
    """Get train accuracy.

    Parameters:
    -----------
        logits: output tensor of model, float - [batch_size, num_classes].
        labels: labels tensor, int32 - [batch_size, num_classes], one hot

    Returns:
    --------
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
    """
    with tf.variable_scope('accuracy') as scope:
<<<<<<< HEAD
        labels_ = tf.argmax(labels, 1)                # transform one hot to labels, [batch_size, ]
=======
        labels_ = tf.argmax(labels, 1)                          # transform one hot to labels, [batch_size, ]
>>>>>>> c80ea2e7674560a2cf52dd7ca54862f4c7379dd0
        top_one = tf.nn.in_top_k(logits, labels_, 1)  # in_top_k
        correct = tf.cast(top_one, tf.float16)
        accuracy = tf.reduce_mean(correct) * 100
        tf.summary.scalar(scope.name+'/accuracy', accuracy)
    return accuracy

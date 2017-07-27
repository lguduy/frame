#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2017/7/22 19:30

@author: liangyu
"""

import os
import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt


NUM_CLASSES = 11


def getDatafile(file_dir, train_size, val_size):
    """Get list of train, val, test image path and label

    Parameters:
    ----------
        file_dir : str, file directory
        train_size : float, size of test set
        val_size : float, size of validation set
        
    Returns:
    -------
        train_img : list of train image path, str
        train_labels : list of train label, int
        test_img :
        test_labels :
        val_img :
        val_labels :
    """

    # images path list
    images_path = []
    # os.walk
    for root, sub_folders, files in os.walk(file_dir):
        for name in files:
            images_path.append(os.path.join(root, name))

    # labels，images path have label of image
    labels = []
    for image_path in images_path:
        label = int(image_path.split('/')[-2])
        labels.append(label)

    # merge image path and label to shuffle
    temp = np.array([images_path, labels]).transpose()
    np.random.shuffle(temp)

    images_path_list = temp[:, 0]    # image path
    labels_list = temp[:, 1]                 # label

    # train val test split
    train_num = math.ceil(len(temp) * train_size)
    val_num = math.ceil(len(temp) * val_size)

    # train img and labels
    train_img = images_path_list[0:train_num]
    train_labels = labels_list[0:train_num]
    train_labels = [int(float(i)) for i in train_labels]

    # val img and labels
    val_img = images_path_list[train_num:train_num+val_num]
    val_labels = labels_list[train_num:train_num+val_num]
    val_labels = [int(float(i)) for i in val_labels]

    # test img and labels
    test_img = images_path_list[train_num+val_num:]
    test_labels = labels_list[train_num+val_num:]
    test_labels = [int(float(i)) for i in test_labels]

    return train_img, train_labels, val_img, val_labels, test_img, test_labels


def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_TFRecord(images, labels, save_dir, name):
    """Convert images and labels to TFRecord file.
    Parameters:
    ----------
        images : list of image path, string
        labels : list of labels, int
        save_dir : the directory to save TFRecord file, e.g.: '/home/liangyu/'
        name : the name of TFRecord file, string, e.g.: 'train'

    Returns:
    --------
        no return
    """

    filename = os.path.join(save_dir, 'cache', name + '.tfrecords')
    n_samples = len(labels)

    if np.shape(images)[0] != n_samples:
        raise ValueError('Images size {} does not match label size {}'.format(images.shape[0], n_samples))

    writer = tf.python_io.TFRecordWriter(filename)       #  TFRecordWriter class
    print '\nTransform start...'
    for i in xrange(0, n_samples):
        try:
            image = plt.imread(images[i])                # type(image) must be array
            image_raw = image.tobytes()                  # transform array to bytes
            label = int(labels[i])
            example = tf.train.Example(features=tf.train.Features(feature={
                            'label': int64_feature(label),
                            'image_raw': bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            print 'Could not read:{}'.format(images[i])
            print 'error: {}'.format(e)
            print 'Skip it!\n'
    writer.close()
    print 'Transform done!'


def read_and_decode(TFRecord_file, batch_size, one_hot, standardize=True):
    """Read and decode TFRecord

    Parameters:
    ----------
        TFRecord_file : filename of TFRecord file, str
        batch_size : batch size, int
        one_hot : label one hot
        standard : Standardize the figure

    Returns:
        image_batch : a batch of image
        label_batch : a batch of label, one hot or not
    -------
    """
    # queue
    filename_queue = tf.train.string_input_producer([TFRecord_file])
    # reader
    reader = tf.TFRecordReader()    
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                       })
    # image decode
    image = tf.decode_raw(features['image_raw'], tf.uint8)    # image tf.uini8
    # image cast
    image = tf.cast(image, tf.float32)                        # image tf.float32

    ############################################################################
    # data augmentation here
    ############################################################################
    
    # reshape
    image = tf.reshape(image, [128, 128, 3])
    # standardization
    if standardize:
        image = tf.image.per_image_standardization(image)

    # label
    label = tf.cast(features['label'], tf.int32)    # label tf.int32
    # image batch, label batch
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size=batch_size,
                                                num_threads=4,
                                                capacity=10000)    # capacity
    
    # one hot
    if one_hot:
        n_classes = NUM_CLASSES
        label_batch = tf.one_hot(label_batch, depth=n_classes)
        label_batch = tf.reshape(label_batch, [batch_size, n_classes])
    else:
        label_batch = tf.reshape(label_batch, [batch_size])
        
    label_batch = tf.cast(label_batch, tf.int32)    # label tf.int32
    
    return image_batch, label_batch


# Main
if __name__ == '__main__':
    # figure dir
    project_dir = os.getcwd()
    figure_dir = os.path.join(project_dir, 'figure')
    
    # get list of images path and list of labels
    train_img, train_labels, val_img, val_labels, test_img, test_labels = getDatafile(figure_dir,
                                                                                      train_size=0.67,
                                                                                      val_size=0.1)
    # convert TFRecord file
    TFRecord_list = ['train', 'val', 'test']
    img_labels_list = [[train_img, train_labels], [val_img, val_labels], [test_img, test_labels]]
    save_dir = os.getcwd()
    for index, TFRecord_name in enumerate(TFRecord_list):
        convert_to_TFRecord(img_labels_list[index][0], img_labels_list[index][1],
                            save_dir,
                            TFRecord_name)

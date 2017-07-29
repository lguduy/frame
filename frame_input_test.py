#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2017/7/23 10:00

frame_input test and plot

@author: liangyu
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from frame.frame_input import read_and_decode


if __name__ == '__main__':
    # test read TFRecord
    BATCH_SIZE = 16
    # file dir
    project_dir = os.getcwd()
    # TFRecord file
    TFRecord_file_list = ['train.tfrecords', 'val.tfrecords', 'test.tfrecords']
    TFRecord_file = os.path.join(project_dir, 'cache', TFRecord_file_list[0])

    # read and decode TFRecord op
    image_batch, label_batch = read_and_decode(TFRecord_file,
                                               batch_size=BATCH_SIZE,
                                               one_hot=True,
                                               standardize=False)

    # test
    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop() and i < 1:
                # run op
                img, label = sess.run([image_batch, label_batch])

                # for visualization
                img_ = np.uint8(img)              # float to uint8
                label_ = np.argmax(label, 1)      # one hot to int

                # just test one batch
                for j in xrange(BATCH_SIZE):
                    print 'label: {}'.format(label_[j])
                    plt.imshow(img_[j, ...])
                    plt.show()
                print "Batch size: {}".format(j+1)
                i+=1

        except tf.errors.OutOfRangeError:
            print 'Done!'
        finally:
            coord.request_stop()
        coord.join(threads)

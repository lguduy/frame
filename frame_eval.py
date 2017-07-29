#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Evaluation the model using test data

@author: liangyu
"""

import tensorflow as tf
import os
import numpy as np

from frame.frame_input import read_and_decode
from frame import frame_model


TEST_BATCH_SIZE = 1        # test batch size
ONE_HOT = True
NUM_CLASSES = 11


def evaluation():
    """Evaluation the model using test data.

    Returns:
    --------
        test_accuracy : test accuracy
    """
    project_dir = os.getcwd()   # project dir
    # tfrecords file
    tfrecord_file = os.path.join(project_dir, 'cache', 'test.tfrecords')
    # checkpoint_dir
    checkpoint_dir = os.path.join(project_dir, 'logs', 'train')
    # test size
    num_test = 2165

    with tf.Graph().as_default():
        test_img_batch, test_label_batch = read_and_decode(tfrecord_file,
                                                           batch_size=TEST_BATCH_SIZE,
                                                           one_hot=ONE_HOT)
        test_logits = frame_model.inference(test_img_batch,
                                            batch_size=TEST_BATCH_SIZE,
                                            n_classes=NUM_CLASSES,
                                            visualize=True)

        test_logits_ = tf.argmax(test_logits, 1)
        test_label_batch_ = tf.argmax(test_label_batch, 1)
        top_one_op = tf.nn.in_top_k(test_logits, test_label_batch_, 1)

        # init saver
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # load saved model
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)    # checkpoint dir
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print 'Checkpoint file has found, global_step: {}'.format(global_step)
            else:
                print 'No checkpoint file found in {}'.format(checkpoint_dir)
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)

            try:
                num_iter = num_test    # batch_size = 1
                true_count = 0
                step = 0

                while step < num_iter and not coord.should_stop():
                    predictions = sess.run([top_one_op])

                    if np.sum(predictions) == 0:
                        print 'true label: {}, predict label: {}'.format(test_label_batch_.eval(),
                                           test_logits_.eval())
                    true_count += np.sum(predictions)
                    step += 1

                test_accuracy = float(true_count) / num_test
                print 'Test accuracy = {:.4f}'.format(test_accuracy)
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)

    return test_accuracy

# Main
if __name__ == '__main__':
    # test
    test_accuracy = evaluation()

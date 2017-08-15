#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Train and save model

@author: liangyu
"""

from __future__ import print_function

import tensorflow as tf
import os
import time

from frame.frame_input import read_and_decode
from frame import frame_model


N_CLASSES = 11
ONE_HOT = True
IMG_W = 128
IMG_H = 128
BATCH_SIZE = 16
MAX_STEP = 100000

learning_rate = 0.0001
visualize = True


def training():
    """Traing model using train iamge and label, validation using val image and label."""
    # project dir
    project_dir = os.getcwd()
    # train data val data path
    train_data_path = os.path.join(project_dir, 'cache', 'train.tfrecords')
    val_data_path = os.path.join(project_dir, 'cache', 'val.tfrecords')

    # logs path
    logs_train_dir = os.path.join(project_dir, 'logs_test_', 'train/')
    logs_val_dir = os.path.join(project_dir, 'logs_test_', 'val/')

    with tf.Graph().as_default():
        # read and decode batch data
        train_img_batch, train_label_batch = read_and_decode(train_data_path,
                                                             batch_size=BATCH_SIZE,
                                                             one_hot=ONE_HOT)
        val_img_batch, val_label_batch = read_and_decode(val_data_path,
                                                         batch_size=BATCH_SIZE,
                                                         one_hot=ONE_HOT)

        # placeholder
        # shape=[NONE, IMG_W, IMG_H, 3] ???
        image = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3], name='image')
        label_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, N_CLASSES], name='label')    # int32

        # model
        logits = frame_model.inference(image, BATCH_SIZE, N_CLASSES, visualize)
        # loss function    train_label_batch y_
        loss = frame_model.losses(logits, label_)
        # optimizer
        optimizer = frame_model.trainning(loss, learning_rate)

        # evaluation
        accuracy = frame_model.evaluation(logits, label_)

        # init op
        init_op = tf.global_variables_initializer()

        # merge summary
        summary_op = tf.summary.merge_all()

        # sess
        with tf.Session() as sess:
            # initial tf.train.Saver() class
            saver = tf.train.Saver()
            # run init op
            sess.run(init_op)
            # start input enqueue threads to read data
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # initial tf.summary.FileWriter() class
            train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
            val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)

            try:
                for step in xrange(MAX_STEP):
                    if coord.should_stop():
                            break

                    # train data batch
                    train_img, train_label = sess.run([train_img_batch, train_label_batch])

                    start_time = time.time()
                    # run ops
                    _, tra_loss, tra_accuracy = sess.run([optimizer, loss, accuracy],
                                                         feed_dict={image: train_img, label_: train_label})

                    duration = time.time() - start_time

                    # print info of training
                    if step % 100 == 0 or (step + 1) == MAX_STEP:
                        sec_per_batch = float(duration)    # training time of a batch
                        print('Step {}, train loss = {:.2f}, train accuracy =',
                        '{:.2f}%, sec_per_batch = {:.2f}s'.format(step,
                                                                 tra_loss,
                                                                 tra_accuracy,
                                                                 sec_per_batch))

                        # run summary op and write train summary to disk
                        summary_str = sess.run(summary_op,
                                               feed_dict={image: train_img, label_: train_label})
                        train_writer.add_summary(summary_str, step)

                    if step % 500 == 0 or (step + 1) == MAX_STEP:
                        # val data batch
                        val_img, val_label = sess.run([val_img_batch, val_label_batch])

                        # run ops
                        val_loss, val_acc, summary_str = sess.run([loss, accuracy, summary_op],
                                                                  feed_dict={image: val_img, label_:val_label})
                        print('*** Step {}, val loss = {:.2f}, val accuracy = {:.2f}% ***'.format(step, val_loss, val_acc))

                        # run summary op and write val summary to disk
                        val_writer.add_summary(summary_str, step)

                    # save model
                    if step % 2000 == 0 or (step + 1) == MAX_STEP:
                        checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)

            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()

            coord.join(threads)


# Main
if __name__ == '__main__':
    start_time = time.time()
    training()
    print('Total time: {:.2f}'.format(time.time() - start_time))

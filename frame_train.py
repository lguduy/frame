#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Train and save model

@author: liangyu
"""

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
    logs_train_dir = os.path.join(project_dir, 'logs', 'train/')
    logs_val_dir = os.path.join(project_dir, 'logs', 'val/')

    with tf.Graph().as_default():
        # read and decode batch data
        train_img_batch, train_label_batch = read_and_decode(train_data_path,
                                                             batch_size=BATCH_SIZE,
                                                             one_hot=ONE_HOT)
        val_img_batch, val_label_batch = read_and_decode(val_data_path,
                                                             batch_size=BATCH_SIZE,
                                                             one_hot=ONE_HOT)
        # model
        logits = frame_model.inference(train_img_batch, BATCH_SIZE, N_CLASSES, visualize)
        # loss function
        loss = frame_model.losses(logits, train_label_batch)
        # optimizer
        optimizer = frame_model.trainning(loss, learning_rate)
        
        # evaluation
        accuracy = frame_model.evaluation(logits, train_label_batch)

        # init op
        init_op = tf.global_variables_initializer()

        # merge summary
        summary_op = tf.summary.merge_all()

        # placeholder
        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
        y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, N_CLASSES])    # int32

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
                                                    feed_dict={x: train_img, y_: train_label})
                    duration = time.time() - start_time
                    
                    # print info of training
                    if step % 100 == 0 or (step + 1) == MAX_STEP:
                        sec_per_batch = float(duration)    # training time of a batch
                        print 'Step {}, train loss = {:.2f}, train accuracy = \
                        {:.2f}%, sec_per_batch = {:.2f}s'.format(step,
                                                                 tra_loss,
                                                                 tra_accuracy,
                                                                 sec_per_batch)

                        # run summary op and write train summary to disk
                        summary_str = sess.run(summary_op)
                        train_writer.add_summary(summary_str, step)

                    if step % 500 == 0 or (step + 1) == MAX_STEP:
                        # validation
                        val_img, val_label = sess.run([val_img_batch, val_label_batch])
                        val_loss, val_acc = sess.run([loss, accuracy],
                                                     feed_dict={x: val_img, y_: val_label})
                        print '*** Step {}, val loss = {:.2f}, val accuracy = {:.2f}% ***'.format(step, val_loss, val_acc)

                        # run summary op and write val summary to disk
                        summary_str = sess.run(summary_op)
                        val_writer.add_summary(summary_str, step)

                    # save model
                    if step % 2000 == 0 or (step + 1) == MAX_STEP:
                        checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)

            except tf.errors.OutOfRangeError:
                print 'Done training -- epoch limit reached'
            finally:
                coord.request_stop()

            coord.join(threads)


# Main
if __name__ == '__main__':
    start_time = time.time()
    training()
    print 'Total time: {:.2f}'.format(time.time() - start_time)

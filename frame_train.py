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
<<<<<<< HEAD

=======
    
>>>>>>> c80ea2e7674560a2cf52dd7ca54862f4c7379dd0
    # batch data
    train_img_batch, train_label_batch = read_and_decode(train_data_path,
                                                         batch_size=BATCH_SIZE,
                                                         one_hot=ONE_HOT)
    val_img_batch, val_label_batch = read_and_decode(val_data_path,
                                                         batch_size=BATCH_SIZE,
                                                         one_hot=ONE_HOT)
    # model
    train_logits = frame_model.inference(train_img_batch, BATCH_SIZE, N_CLASSES, visualize)
    # loss function
    train_loss = frame_model.losses(train_logits, train_label_batch)
    # optimizer
    train_op = frame_model.trainning(train_loss, learning_rate)
    # evaluation
    train_acc = frame_model.evaluation(train_logits, train_label_batch)

    # init op
    init_op = tf.global_variables_initializer()
<<<<<<< HEAD

=======
    
>>>>>>> c80ea2e7674560a2cf52dd7ca54862f4c7379dd0
    # merge summary
    summary_op = tf.summary.merge_all()

    # placeholder
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, N_CLASSES])    # int32

    # sess
    with tf.Session() as sess:
        # save model, weights
        saver = tf.train.Saver()
        # run init op
        sess.run(init_op)
        # start input enqueue threads to read data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # init write logs
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)

        try:
            for step in xrange(MAX_STEP):
                if coord.should_stop():
                        break

                # train data batch
                train_img, train_label = sess.run([train_img_batch, train_label_batch])
<<<<<<< HEAD

=======
                
>>>>>>> c80ea2e7674560a2cf52dd7ca54862f4c7379dd0
                # train optimizers, train loss, train accuracy
                start_time = time.time()
                _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc],
                                                feed_dict={x: train_img, y_: train_label})    # run ops
                duration = time.time() - start_time

                if step % 100 == 0 or (step + 1) == MAX_STEP:
                    sec_per_batch = float(duration)    # training time of a batch
<<<<<<< HEAD
                    print 'Step {}, train loss = {:.2f}, \
                           train accuracy = {:.2f}%, \
                           sec_per_batch = {:.2f}s'.format(step, tra_loss, tra_acc, sec_per_batch)

=======
                    print 'Step %d, train loss = %.2f, train accuracy = %.2f%%, sec_per_batch = %.2fs' %(step,
                                                                                                 tra_loss, tra_acc,
                                                                                                 sec_per_batch)
                    
>>>>>>> c80ea2e7674560a2cf52dd7ca54862f4c7379dd0
                if step % 500 == 0 or (step + 1) == MAX_STEP:
                    # validation
                    val_img, val_label = sess.run([val_img_batch, val_label_batch])
                    val_loss, val_acc = sess.run([train_loss, train_acc],
                                                feed_dict={x: val_img, y_: val_label})
                    print '*** Step {}, val loss = {:.2f}, val accuracy = {:.2f}% ***'.format(step, val_loss, val_acc)

                    # write train logs for tensorboard
                    summary_str = sess.run(summary_op)    # run summary op
                    train_writer.add_summary(summary_str, step)
                    # write val logs for tensorboard
                    summary_str = sess.run(summary_op)
                    val_writer.add_summary(summary_str, step)

                # save model
                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

        except tf.errors.OutOfRangeError:
            print 'Done training, epoch limit reached'
        finally:
            coord.request_stop()
<<<<<<< HEAD

=======
            
>>>>>>> c80ea2e7674560a2cf52dd7ca54862f4c7379dd0
        coord.join(threads)


# Main
if __name__ == '__main__':
    start_time = time.time()
    training()
    print 'Total time: {:.2f}'.format(time.time() - start_time)

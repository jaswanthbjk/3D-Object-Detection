''' Training Frustum PointNets.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import argparse
import importlib
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from datetime import datetime
from absl import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import provider
from train_util import get_batch
from prepare_lyft_data_v2 import parse_data
from run_prepare_lyft_data import list_all_files

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--restore_model_path', default=None, help='Restore model path e.g. log/model.ckpt [default: None]')
parser.add_argument('--data_dir', default=None, help="overwritten data path for provider.FrustumData")
FLAGS = parser.parse_args()

logging.set_verbosity(logging.INFO)


import model_util

# Set training configurations
EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = model_util.NUM_POINTS_OF_PC
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
NUM_CHANNEL = model_util.NUM_CHANNELS_OF_PC
NUM_CLASSES = 2  # segmentation has two classes

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'Reference', FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp %s %s' % (os.path.join(BASE_DIR, 'train.py'), LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99



# Load Frustum Datasets. Use default data paths.
# TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='val',
#                                       rotate_to_center=True, one_hot=True, overwritten_data_path=FLAGS.data_dir)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.compat.v1.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.compat.v1.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    """ Main function for training and simple evaluation. """
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            filenames = list_all_files(FLAGS.data_dir)
            full_dataset = tf.compat.v1.data.TFRecordDataset(filenames)
            parsed_dataset = full_dataset.map(parse_data)
            parsed_dataset = parsed_dataset.repeat(MAX_EPOCH).batch(BATCH_SIZE)

            # iterator = parsed_dataset.make_one_shot_iterator()
            iterator = tf.compat.v1.data.make_one_shot_iterator(parsed_dataset)

            pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
            heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl = \
                iterator.get_next()

            tf.ensure_shape(pointclouds_pl, (BATCH_SIZE, NUM_POINT, NUM_CHANNEL))

            is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())

            # Note the global_step=batch_size parameter to minimize.
            # That tells the optimizer to increment the 'batch_size' parameter
            # for you every time it trains.
            batch = tf.compat.v1.get_variable('batch_size', [],
                                    initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.compat.v1.summary.scalar('bn_decay', bn_decay)

            # Get model and losses 
            end_points = MODEL.get_model(pointclouds_pl, one_hot_vec_pl,
                                         is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(labels_pl, centers_pl,
                                  heading_class_label_pl, heading_residual_label_pl,
                                  size_class_label_pl, size_residual_label_pl, end_points)
            tf.compat.v1.summary.scalar('loss', loss)

            losses = tf.compat.v1.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.compat.v1.summary.scalar('total_loss', total_loss)

            # Write summaries of bounding box IoU and segmentation accuracies
            iou2ds, iou3ds = tf.py_func(provider.compute_box3d_iou, [ \
                end_points['center'], \
                end_points['heading_scores'], end_points['heading_residuals'], \
                end_points['size_scores'], end_points['size_residuals'], \
                centers_pl, \
                heading_class_label_pl, heading_residual_label_pl, \
                size_class_label_pl, size_residual_label_pl], \
                                                  [tf.float32, tf.float32])
            # iou2ds = iou2ds.numpy()
            # iou3ds = iou3ds.numpy()
            end_points['iou2ds'] = iou2ds
            end_points['iou3ds'] = iou3ds
            tf.compat.v1.summary.scalar('iou_2d', tf.reduce_mean(iou2ds))
            tf.compat.v1.summary.scalar('iou_3d', tf.reduce_mean(iou3ds))

            correct = tf.equal(tf.argmax(end_points['mask_logits'], 2),
                               tf.cast(labels_pl,dtype=tf.int64))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / \
                       float(BATCH_SIZE * NUM_POINT)
            tf.compat.v1.summary.scalar('segmentation accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.compat.v1.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate,
                                                       momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.compat.v1.train.Saver()

        # Create a session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.compat.v1.Session(config=config)

        # Add summary writers
        merged = tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        if FLAGS.restore_model_path is None:
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
        else:
            saver.restore(sess, FLAGS.restore_model_path)

        ops = {'pointclouds_pl': pointclouds_pl,
               'one_hot_vec_pl': one_hot_vec_pl,
               'labels_pl': labels_pl,
               'centers_pl': centers_pl,
               'heading_class_label_pl': heading_class_label_pl,
               'heading_residual_label_pl': heading_residual_label_pl,
               'size_class_label_pl': size_class_label_pl,
               'size_residual_label_pl': size_residual_label_pl,
               'is_training_pl': is_training_pl,
               'logits': end_points['mask_logits'],
               'centers_pred': end_points['center'],
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        log_string('**** START TRAINING ****')
        sys.stdout.flush()

        train_one_epoch(sess, ops, train_writer, model_saver=saver)
        # eval_one_epoch(sess, ops, test_writer)


def train_one_epoch(sess, ops, train_writer, model_saver):
    ''' Training for one epoch on the frustum dataset.
    ops is dict mapping from string to tf ops
    '''
    is_training = True
    log_string(str(datetime.now()))

    import itertools

    # To collect statistics
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    iou2ds_sum = 0
    iou3ds_sum = 0
    iou3d_correct_cnt = 0

    for count_num in itertools.count(start=1):
        try:
            print("I am here")
            summary, step, _, loss_val, logits_val, centers_pred_val, \
            iou2ds, iou3ds, batch_label = \
                sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'],
                          ops['logits'], ops['centers_pred'],
                          ops['end_points']['iou2ds'], ops['end_points']['iou3ds'], ops['labels_pl']],
                         feed_dict={ops['is_training_pl']: is_training})

            logging.debug("Number of batch: {}".format(count_num))

            train_writer.add_summary(summary, step)

            preds_val = np.argmax(logits_val, 2)
            correct = np.sum(preds_val == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss_val
            iou2ds_sum += np.sum(iou2ds)
            iou3ds_sum += np.sum(iou3ds)
            iou3d_correct_cnt += np.sum(iou3ds >= 0.7)

            if count_num % 20 == 0:
                # if (batch_idx + 1) % 10 == 0:
                #   log_string(' -- %03d / %03d --' % (batch_idx + 1, num_batches))
                log_string('mean loss: %f' % (loss_sum / (count_num * BATCH_SIZE)))
                log_string('segmentation accuracy: %f' % \
                           (total_correct / float(total_seen)))
                log_string('box IoU (ground/3D): %f / %f' % \
                           (iou2ds_sum / float(BATCH_SIZE * count_num), iou3ds_sum / float(BATCH_SIZE * count_num)))
                log_string('box estimation accuracy (IoU=0.7): %f' % \
                           (float(iou3d_correct_cnt) / float(BATCH_SIZE * count_num)))

                # Save the variables to disk.
                save_path = model_saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)
        except tf.errors.OutOfRangeError:
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            iou2ds_sum = 0
            iou3ds_sum = 0
            iou3d_correct_cnt = 0
            break


def eval_one_epoch(sess, ops, test_writer):
    ''' Simple evaluation for one epoch on the frustum dataset.
    ops is dict mapping from string to tf ops """
    '''
    global EPOCH_CNT
    is_training = False
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----' % (EPOCH_CNT))
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = int(len(TEST_DATASET) / BATCH_SIZE)

    # To collect statistics
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    iou2ds_sum = 0
    iou3ds_sum = 0
    iou3d_correct_cnt = 0

    # Simple evaluation with batches 
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        batch_data, batch_label, batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres, \
        batch_rot_angle, batch_one_hot_vec = \
            get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,
                      NUM_POINT, NUM_CHANNEL)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['one_hot_vec_pl']: batch_one_hot_vec,
                     ops['labels_pl']: batch_label,
                     ops['centers_pl']: batch_center,
                     ops['heading_class_label_pl']: batch_hclass,
                     ops['heading_residual_label_pl']: batch_hres,
                     ops['size_class_label_pl']: batch_sclass,
                     ops['size_residual_label_pl']: batch_sres,
                     ops['is_training_pl']: is_training}

        summary, step, loss_val, logits_val, iou2ds, iou3ds = \
            sess.run([ops['merged'], ops['step'],
                      ops['loss'], ops['logits'],
                      ops['end_points']['iou2ds'], ops['end_points']['iou3ds']],
                     feed_dict=feed_dict)
        test_writer.add_summary(summary, step)

        preds_val = np.argmax(logits_val, 2)
        correct = np.sum(preds_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE * NUM_POINT)
        loss_sum += loss_val
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum(batch_label == l)
            total_correct_class[l] += (np.sum((preds_val == l) & (batch_label == l)))
        iou2ds_sum += np.sum(iou2ds)
        iou3ds_sum += np.sum(iou3ds)
        iou3d_correct_cnt += np.sum(iou3ds >= 0.7)

        for i in range(BATCH_SIZE):
            segp = preds_val[i, :]
            segl = batch_label[i, :]
            part_ious = [0.0 for _ in range(NUM_CLASSES)]
            for l in range(NUM_CLASSES):
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
                    part_ious[l] = 1.0  # class not present
                else:
                    part_ious[l] = np.sum((segl == l) & (segp == l)) / \
                                   float(np.sum((segl == l) | (segp == l)))

    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval segmentation accuracy: %f' % \
               (total_correct / float(total_seen)))
    log_string('eval segmentation avg class acc: %f' % \
               (np.mean(np.array(total_correct_class) / \
                        np.array(total_seen_class, dtype=np.float))))
    log_string('eval box IoU (ground/3D): %f / %f' % \
               (iou2ds_sum / float(num_batches * BATCH_SIZE), iou3ds_sum / \
                float(num_batches * BATCH_SIZE)))
    log_string('eval box estimation accuracy (IoU=0.7): %f' % \
               (float(iou3d_correct_cnt) / float(num_batches * BATCH_SIZE)))

    EPOCH_CNT += 1


if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    train()
    LOG_FOUT.close()

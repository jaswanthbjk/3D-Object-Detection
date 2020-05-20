''' Evaluating Frustum PointNets.

Adapted from test.py of frustum pointnet
'''
from __future__ import print_function

import os
import sys
import argparse
import importlib
import numpy as np
import tensorflow as tf
import pickle
import math
import itertools

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from utils.model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
from dataset import provider
from train.train_util import get_batch

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
parser.add_argument('--model_path', default='log_v1_test/model.ckpt',
                    help='model checkpoint file path [default: ./log_v1_test/model.ckpt]')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size size for inference [default: 32]')
parser.add_argument('--output', default='test_results', help='output file/folder name [default: test_results]')
parser.add_argument('--data_path', default=None, help='frustum dataset pickle filepath [default: None]')
parser.add_argument('--from_rgb_detection', action='store_true', help='test from dataset files from rgb detection.')
parser.add_argument('--idx_path', default=None,
                    help='filename of txt where each line is a data idx, used for rgb detection -- write <id>.txt for all frames. [default: None]')
parser.add_argument('--dump_result', action='store_true', help='If true, also dump results to .pickle file')
parser.add_argument('--no_intensity', action='store_true', help='Only use XYZ for training')
parser.add_argument('--data_dir', default=None, help="overwritten data path for provider.FrustumData")
parser.add_argument('--file_pattern', default="scene_\d_xxx.tfrec", help="file patter (regular expression)")
FLAGS = parser.parse_args()

from utils import model_util

# Set training configurations
BATCH_SIZE = FLAGS.batch_size
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
MODEL = importlib.import_module(FLAGS.model)
NUM_CLASSES = 2
NUM_CHANNEL = model_util.NUM_CHANNELS_OF_PC
print('The Data directory is {}'.format(FLAGS.data_dir))

from dataset.prepare_lyft_data_v2 import parse_inference_data, \
    get_inference_results_tfexample
from dataset.run_prepare_lyft_data import list_all_files
print("The data directory is {}".format(FLAGS.data_dir))


def get_session_and_ops():
    ''' Define model graph, load model parameters,
    create session and return session handle and tensors
    '''
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            filenames = list_all_files(FLAGS.data_dir, pat=FLAGS.file_pattern)
            print(filenames)
            full_dataset = tf.data.TFRecordDataset(filenames)
            parsed_dataset = full_dataset.map(parse_inference_data)
            parsed_dataset = parsed_dataset.batch(BATCH_SIZE)

            iterator = parsed_dataset.make_one_shot_iterator()

            pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
            heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl, \
            sample_token_tensor, camera_token_tensor, frustum_angle_tensor,type_name_tensor = \
                iterator.get_next()

            tf.ensure_shape(pointclouds_pl, (BATCH_SIZE, NUM_POINT, NUM_CHANNEL))

            is_training_pl = tf.placeholder(tf.bool, shape=())
            end_points = MODEL.get_model(pointclouds_pl, one_hot_vec_pl,
                                         is_training_pl)
            loss = MODEL.get_loss(labels_pl, centers_pl,
                                  heading_class_label_pl, heading_residual_label_pl,
                                  size_class_label_pl, size_residual_label_pl, end_points)
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
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
               'center': end_points['center'],
               'end_points': end_points,
               'loss': loss,
               'camera_token': camera_token_tensor,
               'sample_token': sample_token_tensor,
               'frustum_angle': frustum_angle_tensor,
               'type_name': type_name_tensor}
        return sess, ops


def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs


def inference_new(sess, ops):
    feed_dict = {ops['is_training_pl']: False}
    np.seterr(divide='ignore', invalid='ignore')
    ep = ops['end_points']
    from helpers.config_tool import get_paths
    _, artifact_path, _ = get_paths()
    with tf.io.TFRecordWriter(os.path.join(artifact_path, "inference_results_v2.tfrec")) as tfrw:
        for count_num in itertools.count():
            try:
                print('sample number is {}'.format(count_num))
                batch_logits, batch_centers, \
                batch_heading_scores, batch_heading_residuals, \
                batch_size_scores, batch_size_residuals, \
                camera_token_bytes_string, sample_token_bytes_string, \
                point_clouds, seg_labels, frustum_angle, type_name_bytes_string = \
                    sess.run([ops['logits'], ops['center'],
                              ep['heading_scores'], ep['heading_residuals'],
                              ep['size_scores'], ep['size_residuals'],
                              ops['camera_token'], ops['sample_token'],
                              ops['pointclouds_pl'], ops['labels_pl'],
                              ops['frustum_angle'],ops['type_name']],
                             feed_dict=feed_dict)

                # Compute scores
                batch_seg_prob = softmax(batch_logits)[:, :, 1]  # BxN
                batch_seg_mask = np.argmax(batch_logits, 2)  # BxN
                mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1)  # B,
                mask_mean_prob = mask_mean_prob / np.sum(batch_seg_mask, 1)  # B,
                heading_prob = np.max(softmax(batch_heading_scores), 1)  # B
                size_prob = np.max(softmax(batch_size_scores), 1)  # B,

                # batch_size score includes the score of segmentation mask accuracy, heading, and size
                batch_scores = np.log(mask_mean_prob) + np.log(heading_prob) + np.log(size_prob)
                batch_prob=(mask_mean_prob+heading_prob+size_prob)/3

                heading_cls = np.argmax(softmax(batch_heading_scores), 1)  # B
                size_cls = np.argmax(softmax(batch_size_scores), 1)  # B

                current_batch_size = batch_logits.shape[0]
                for batch_index in range(current_batch_size):
                    heading_res = batch_heading_residuals[batch_index, heading_cls[batch_index]]
                    size_res = batch_size_residuals[batch_index, size_cls[batch_index], :]

                    example_msg = get_inference_results_tfexample(point_cloud=point_clouds[batch_index, ...],
                                                                  seg_label=seg_labels[batch_index, ...],
                                                                  seg_label_logits=batch_logits[batch_index, ...],
                                                                  box_center=batch_centers[batch_index, ...],
                                                                  heading_angle_class=heading_cls[batch_index, ...],
                                                                  heading_angle_residual=heading_res,
                                                                  size_class=size_cls[batch_index, ...],
                                                                  size_residual=size_res,
                                                                  frustum_angle=frustum_angle[batch_index, ...],
                                                                  score=batch_prob[batch_index, ...],
                                                                  camera_token=camera_token_bytes_string[
                                                                      batch_index].decode('utf8'),
                                                                  sample_token=sample_token_bytes_string[
                                                                      batch_index].decode('utf8'),
                                                                  type_name=type_name_bytes_string[batch_index].decode('utf8'))

                    tfrw.write(example_msg.SerializeToString())

                # Finished computing scores
            except tf.errors.OutOfRangeError:
                break


def test_new():
    sess, ops = get_session_and_ops()
    inference_new(sess, ops)


if __name__ == '__main__':
    test_new()

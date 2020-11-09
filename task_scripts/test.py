import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam
import tensorflow.python.keras.backend as K
from tensorflow.python import keras
from tensorflow.keras.models import model_from_json

import os
import numpy as np
import json
import pickle
import time
import itertools
from datetime import datetime

from FPointNet_Keras import Frustum_Pointnet_Model
from model_util import parse_test_data
from provider import FrustumDataset
import provider

NUM_EPOCHS = 200
NUM_POINT = 1024
NUM_CHANNEL = 3
BATCH_SIZE = 1
MAX_EPOCH = 10

NUM_CLASSES = 2
NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8  # one cluster for each type
NUM_OBJECT_POINT = 512

g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
g_onehotclass2type = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}


# model.summary()

def softmax(x):
    """ Numpy function for softmax"""
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs


LOG_DIR = './log_test/'
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_test.txt'), 'w')


def write_detection_results(result_dir, id_list, type_list, box2d_list, center_list,
                            heading_cls_list, heading_res_list,
                            size_cls_list, size_res_list,
                            rot_angle_list, score_list):
    """ Write frustum pointnets results to KITTI format label files. """
    if result_dir is None:
        return
    results = {}  # map from idx to list of strings, each string is a line (without \n)
    for i in range(len(center_list)):
        idx = id_list[i]
        output_str = g_onehotclass2type[type_list[i]] + " -1 -1 -10 "
        box2d = box2d_list[i]
        output_str += "%f %f %f %f " % (box2d[0], box2d[1], box2d[2], box2d[3])
        h, w, l, tx, ty, tz, ry = provider.from_prediction_to_label_format(center_list[i],
                                                                           heading_cls_list[i], heading_res_list[i],
                                                                           size_cls_list[i], size_res_list[i],
                                                                           rot_angle_list[i])
        score = score_list[i]
        output_str += "%f %f %f %f %f %f %f %f" % (h, w, l, tx, ty, tz, ry, score)
        if idx not in results:
            results[idx] = []
        results[idx].append(output_str)

    # Write TXT files
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt' % (idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line + '\n')
        fout.close()


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def fill_files(output_dir, to_fill_filename_list):
    """ Create empty files if not exist for the filelist. """
    for filename in to_fill_filename_list:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            fout = open(filepath, 'w')
            fout.close()


def make_test_iterator(dataset):
    iterator = dataset.make_one_shot_iterator()
    next_val = iterator.get_next()

    with K.get_session().as_default() as sess:
        while True:
            pointclouds_pl, one_hot_vec_pl, rot_angle, prob_pl, class_pl, tokens_pl, box2d_pl = sess.run(next_val)
            data_dict = {"frustum_point_cloud": pointclouds_pl, "one_hot_vec": one_hot_vec_pl, "rot_angle": rot_angle,
                         "rgb_prob": prob_pl, "cls_type": class_pl, "token": tokens_pl,
                         "box_2D": box2d_pl}

            yield data_dict


def inference(feed_dict, model, batch_size):
    pc = feed_dict["frustum_point_cloud"]
    num_batches = int(pc.shape[0] / batch_size)
    logits = np.zeros((pc.shape[0], pc.shape[1], NUM_CLASSES))
    centers = np.zeros((pc.shape[0], 3))
    heading_logits = np.zeros((pc.shape[0], NUM_HEADING_BIN))
    heading_residuals = np.zeros((pc.shape[0], NUM_HEADING_BIN))
    size_logits = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER))
    size_residuals = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER, 3))
    scores = np.zeros((pc.shape[0],))  # 3D box score

    for i in range(num_batches):
        batch_logits, batch_centers, batch_heading_scores, batch_heading_residuals, \
        batch_size_scores, batch_size_residuals = model.predict(feed_dict)

        logits[i * batch_size:(i + 1) * batch_size, ...] = batch_logits
        centers[i * batch_size:(i + 1) * batch_size, ...] = batch_centers
        heading_logits[i * batch_size:(i + 1) * batch_size, ...] = batch_heading_scores
        heading_residuals[i * batch_size:(i + 1) * batch_size, ...] = batch_heading_residuals
        size_logits[i * batch_size:(i + 1) * batch_size, ...] = batch_size_scores
        size_residuals[i * batch_size:(i + 1) * batch_size, ...] = batch_size_residuals

        # Compute scores
        batch_seg_prob = softmax(batch_logits)[:, :, 1]  # BxN
        batch_seg_mask = np.argmax(batch_logits, 2)  # BxN
        mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1)  # B,
        mask_mean_prob = mask_mean_prob / np.sum(batch_seg_mask, 1)  # B,
        heading_prob = np.max(softmax(batch_heading_scores), 1)  # B
        size_prob = np.max(softmax(batch_size_scores), 1)  # B,
        # batch_scores = np.log(mask_mean_prob) + np.log(heading_prob) + np.log(size_prob)
        batch_scores = (mask_mean_prob+heading_prob+size_prob)/3
        print(mask_mean_prob, heading_prob, size_prob, batch_scores)
        scores[i * batch_size:(i + 1) * batch_size] = batch_scores
        # Finished computing scores

    heading_cls = np.argmax(heading_logits, 1)  # B
    size_cls = np.argmax(size_logits, 1)  # B
    heading_res = np.array([heading_residuals[i, heading_cls[i]] for i in range(pc.shape[0])])
    size_res = np.vstack([size_residuals[i, size_cls[i], :] for i in range(pc.shape[0])])

    return np.argmax(logits, 2), centers, heading_cls, heading_res, size_cls, size_res, scores


def test(model, test_loader, output_filename, result_dir=None, idx_path=None):
    ps_list = []
    segp_list = []
    center_list = []
    heading_cls_list = []
    heading_res_list = []
    size_cls_list = []
    size_res_list = []
    rot_angle_list = []
    score_list = []
    onehot_list = []
    class_list = []
    token_list = []
    box2d_list = []

    print(len(TEST_DATASET))
    batch_size = BATCH_SIZE

    for batch_idx, data_dict in enumerate(test_loader):
        print(batch_idx)
        point_cloud = data_dict["frustum_point_cloud"]
        one_hot_vec = data_dict["one_hot_vec"]
        rot_angle = data_dict["rot_angle"]
        prob = data_dict["rgb_prob"]
        cls_type = data_dict["cls_type"]
        token = data_dict["token"]
        box_2D = data_dict["box_2D"]

        # print('batch idx: %s' % (batch_idx))
        start_idx = batch_idx * batch_size
        end_idx = min(len(TEST_DATASET), (batch_idx + 1) * batch_size)
        cur_batch_size = end_idx - start_idx

        X_test = {"frustum_point_cloud": point_cloud, "one_hot_vec": one_hot_vec}

        batch_output, batch_center_pred, batch_hclass_pred, batch_hres_pred, \
        batch_sclass_pred, batch_sres_pred, batch_scores = inference(X_test, model, batch_size)

        for i in range(cur_batch_size):
            ps_list.append(point_cloud[i, ...])
            segp_list.append(batch_output[i, ...])
            center_list.append(batch_center_pred[i, :])
            heading_cls_list.append(batch_hclass_pred[i])
            heading_res_list.append(batch_hres_pred[i])
            size_cls_list.append(batch_sclass_pred[i])
            size_res_list.append(batch_sres_pred[i, :])
            rot_angle_list.append(rot_angle[i])
            score_list.append(batch_scores[i])
            # score_list.append(prob[i])
            onehot_list.append(one_hot_vec[i])
            class_list.append(cls_type[i])
            token_list.append(token[i])
            box2d_list.append(box_2D[i])

        if batch_idx == 25344:
            print("Exhausted the dataset")
            break

    with open(output_filename, 'wb') as fp:
        pickle.dump(ps_list, fp)
        pickle.dump(segp_list, fp)
        pickle.dump(center_list, fp)
        pickle.dump(heading_cls_list, fp)
        pickle.dump(heading_res_list, fp)
        pickle.dump(size_cls_list, fp)
        pickle.dump(size_res_list, fp)
        pickle.dump(rot_angle_list, fp)
        pickle.dump(score_list, fp)
        pickle.dump(onehot_list, fp)
        pickle.dump(class_list, fp)
        pickle.dump(token_list, fp)
        pickle.dump(box2d_list, fp)

    print('Number of point clouds: %d' % (len(ps_list)))

    write_detection_results(result_dir, token_list, class_list, box2d_list, center_list, heading_cls_list,
                            heading_res_list, size_cls_list, size_res_list, rot_angle_list, score_list)
    # Make sure for each frame (no matter if we have measurment for that frame),
    # there is a TXT file
    output_dir = os.path.join(result_dir, 'data')
    if idx_path is not None:
        to_fill_filename_list = [line.rstrip() + '.txt' for line in open(idx_path)]
        fill_files(output_dir, to_fill_filename_list)


if __name__ == '__main__':
    # checkpoints = ['cp-55-0180.ckpt','cp-55-0180.ckpt','cp-60-0180.ckpt','cp-60-0180.ckpt','cp-65-0180.ckpt',
    # 'cp-65-0180.ckpt','cp-70-0180.ckpt','cp-70-0180.ckpt','cp-75-0180.ckpt','cp-75-0180.ckpt','cp-80-0180.ckpt','cp-80-0180.ckpt']
    # checkpoints = ['cp-55-0180','cp-55-0180','cp-60-0180','cp-60-0180','cp-65-0180',
    # 'cp-65-0180','cp-70-0180','cp-70-0180','cp-75-0180','cp-75-0180','cp-80-0180','cp-80-0180']
    checkpoints = ['cp-80-0180']

    idx_path = '/home/jbandl2s/sub_ensembles/kitti/image_sets/val.txt'
    checkpoint_path = "/scratch/jbandl2s/kitti/training/checkpoints/"
    json_path = "/home/jbandl2s/sub_ensembles/models/training_2/model.json"

    TEST_DATASET = FrustumDataset(npoints=NUM_POINT, split='val',
                                  rotate_to_center=True,
                                  overwritten_data_path='/scratch/jbandl2s/kitti/frustum_carpedcyc_val_rgb_detection'
                                                        '.pickle',
                                  from_rgb_detection=True, one_hot=True)

    test_dataset = tf.data.TFRecordDataset('/scratch/jbandl2s/kitti/kitti_test.tfrec')
    parsed_test_dataset = test_dataset.map(parse_test_data)
    parsed_test_dataset = parsed_test_dataset.batch(BATCH_SIZE, drop_remainder=True)



    # save_file_name = os.path.join('./detections', 'detection.pkl')
    # result_folder = os.path.join('./detections', 'result')

    for i, ckpt in enumerate(checkpoints):
        itr_test = make_test_iterator(parsed_test_dataset)
        print('Model with checkpoint {}'.format(ckpt))
        ckpt_path = os.path.join(checkpoint_path,ckpt)
        save_file_name = os.path.join('./detections', 'check_detection_{}.pkl'.format(i))
        result_folder = os.path.join('./detections', 'check_result_{}'.format(i))
        print(result_folder)
        ckpt_path = '/scratch/jbandl2s/lyft_kitti/train/model_ckpts/cp-0081.ckpt'
        _, Fpointnet_Inference_model = Frustum_Pointnet_Model(batch_size=BATCH_SIZE)
        Fpointnet_Inference_model.load_weights(ckpt_path)
        test(model=Fpointnet_Inference_model, test_loader=itr_test, output_filename=save_file_name,
             result_dir=result_folder, idx_path=idx_path)

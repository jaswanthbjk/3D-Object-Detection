import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python import keras
from tensorflow.python.keras.layers import Lambda
import tensorflow.python.keras.backend as K

from FPointNet_Keras import Frustum_Pointnet_Model
from model_util import parse_data

NUM_EPOCHS = 200

import numpy as np
import math
from datetime import datetime
from provider import FrustumDataset, compute_box3d_iou
from train_util import get_batch

NUM_POINT = 1024
NUM_CHANNEL = 3
BATCH_SIZE = 32
MAX_EPOCH = 10

TRAIN_DATASET = FrustumDataset(npoints=NUM_POINT, split='train',
                               rotate_to_center=True, random_flip=True, random_shift=True, one_hot=True)
VAL_DATASET = FrustumDataset(npoints=NUM_POINT, split='val',
                             rotate_to_center=True, one_hot=True)
train_idxs = np.arange(0, len(TRAIN_DATASET))
np.random.shuffle(train_idxs)
num_batches = int(len(TRAIN_DATASET) / BATCH_SIZE)
num_val_batches = int(len(VAL_DATASET) / BATCH_SIZE)

checkpoint_path = "model_ckpts/cp-{epoch:04d}"
checkpoint_dir = os.path.dirname(checkpoint_path)


def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def make_iterator(dataset):
    iterator = dataset.make_one_shot_iterator()
    next_val = iterator.get_next()

    with K.get_session().as_default() as sess:
        while True:
            pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl = sess.run(next_val)

            x_trains = {"frustum_point_cloud": pointclouds_pl, "one_hot_vec": one_hot_vec_pl, "seg_label": labels_pl,
                        "box3d_center": centers_pl, "angle_class": heading_class_label_pl,
                        "size_class": size_class_label_pl, "angle_residual": heading_residual_label_pl,
                        "size_residual": size_residual_label_pl}

            yield x_trains


class Metrics(keras.callbacks.Callback):
    def __init__(self, val_data, step, batch_size=20):
        self.validation_data = val_data
        self.batch_size = batch_size
        self.validation_step = step
        self.total_correct = 0
        self.total_seen = 0
        self.loss_sum = 0
        self.iou2ds_sum = 0.0
        self.iou3ds_sum = 0.0
        self.iou3d_correct_cnt = 0
        print('validation_step  ' + str(step))

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self._iou2d = []
        self._iou3d = []
        self._accdata = []

    def on_epoch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        for batch_index in range(self.validation_step):
            xVal, yVal = next(self.validation_data)
            batch_logits, batch_centers, batch_heading_scores, batch_heading_residuals, \
            batch_size_scores, batch_size_residuals = self.model.predict(xVal)
            labels_pl, centers_pl, heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl = yVal['InsSeg_out'], yVal['center_out'], \
                                                          yVal['heading_scores'], yVal['heading_residual_label'], \
                                                          yVal['size_scores'], yVal['size_residual_label']
            iou2ds, iou3ds = tf.py_func(compute_box3d_iou, [batch_centers, batch_heading_scores,
                                                            batch_heading_residuals, batch_size_scores,
                                                            batch_size_residuals,
                                                            centers_pl,
                                                            heading_class_label_pl, heading_residual_label_pl,
                                                            size_class_label_pl, size_residual_label_pl],
                                        [tf.float32, tf.float32])

            preds_val = np.argmax(batch_logits, 2)
            correct = np.sum(preds_val == labels_pl)
            self.total_correct = Lambda(lambda a: a[0] + a[1])([self.total_correct, correct])
            self.total_seen = Lambda(lambda a: a[0] + a[1])([self.total_seen, (BATCH_SIZE * NUM_POINT)])
            self.iou2ds_sum = Lambda(lambda a: a[0] + a[1])([self.iou2ds_sum, K.sum(iou2ds)])
            self.iou3ds_sum = Lambda(lambda a: a[0] + a[1])([self.iou2ds_sum, K.sum(iou3ds)])
            self.iou3d_correct_cnt = Lambda(lambda a: a[0] + a[1])([self.iou3d_correct_cnt,
                                                                    K.sum(K.cast(iou3ds >= 0.7, tf.int32))])

            self.total_correct = self.total_correct.eval(session=tf.compat.v1.Session())
            self.total_seen = self.total_seen.eval(session=tf.compat.v1.Session())
            self.iou2ds_sum = self.iou2ds_sum.eval(session=tf.compat.v1.Session())
            self.iou3ds_sum = self.iou3ds_sum.eval(session=tf.compat.v1.Session())
            self.iou3d_correct_cnt = self.iou3d_correct_cnt.eval(session=tf.compat.v1.Session())

            print('segmentation accuracy: %f' % \
                  (self.total_correct / float(self.total_seen)))
            print('box IoU (ground/3D): %f / %f' % \
                  (self.iou2ds_sum / float(BATCH_SIZE), self.iou3ds_sum / float(BATCH_SIZE)))
            print('box estimation accuracy (IoU=0.7): %f' % \
                  (float(self.iou3d_correct_cnt) / float(BATCH_SIZE)))

            return


if __name__ == '__main__':
    model, _ = Frustum_Pointnet_Model()  # Using the training model
    model.save("my_model.h5")

    lrate = LearningRateScheduler(step_decay)

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True,
                                                     period=5)
    log_path = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.dirname(log_path)
    tensorboard_callback = tf.python.keras.callbacks.TensorBoard(log_dir=logdir)

    model.compile(optimizer=Adam(lr=1e-3), loss={'fp_loss': lambda y_true, y_pred: y_pred})

    # Using the tfrec files helps us in making the iterator and take advantage of the data loader API from keras
    # The dataset is repeated for 200 epochs. and divided into a batch size of 32
    train_dataset = tf.data.TFRecordDataset('/scratch/jbandl2s/Lyft_dataset/train/lyft_train.tfrec')
    parsed_train_dataset = train_dataset.map(parse_data)
    parsed_train_dataset = parsed_train_dataset.repeat(200).batch(BATCH_SIZE, drop_remainder=True)

    val_dataset = tf.data.TFRecordDataset('/scratch/jbandl2s/Lyft_dataset/train/lyft_val.tfrec')
    parsed_val_dataset = val_dataset.map(parse_data)
    parsed_val_dataset = parsed_val_dataset.repeat(200).batch(BATCH_SIZE, drop_remainder=True)

    itr_train = make_iterator(parsed_train_dataset)
    itr_valid = make_iterator(parsed_val_dataset)

    metrics = Metrics(val_data=itr_valid, step=487, batch_size=32)
    model.save_weights(checkpoint_path.format(epoch=0))  # Initial checkpoint weights storage
    model.fit_generator(generator=itr_train, validation_data=itr_valid, validation_steps=487, epochs=180,
                        steps_per_epoch=2296, callbacks=[cp_callback, tensorboard_callback], verbose=1, workers=0)
    model.save('saved_model/my_model')

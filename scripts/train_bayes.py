import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam
import tensorflow.python.keras.backend as K

# from keras.optimizers import Adam, SGD
from frustum_pointnets_bayes import Frustum_Pointnet_Model
from model_util import parse_data

NUM_EPOCHS = 200

import os
import numpy as np
from datetime import datetime
from provider import FrustumDataset
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

model = Frustum_Pointnet_Model()

loss_funcs = {"InsSeg_out": "sparse_categorical_crossentropy",
              "center_out": "huber_loss",
              "heading_scores": "sparse_categorical_crossentropy",
              "size_scores": "sparse_categorical_crossentropy"}
loss_weights = {"InsSeg_out": 1.0, "center_out": 1.0, "heading_scores": 1.0, "size_scores": 1.0}

model.compile(optimizer=Adam(1e-04), loss=loss_funcs, loss_weights=loss_weights)

train_dataset = tf.data.TFRecordDataset('/scratch/jbandl2s/kitti/kitti_train.tfrec')
parsed_train_dataset = train_dataset.map(parse_data)
parsed_train_dataset = parsed_train_dataset.repeat(200).batch(BATCH_SIZE, drop_remainder=True)

val_dataset = tf.data.TFRecordDataset('/scratch/jbandl2s/kitti/kitti_val.tfrec')
parsed_val_dataset = val_dataset.map(parse_data)
parsed_val_dataset = parsed_val_dataset.repeat(200).batch(BATCH_SIZE, drop_remainder=True)

checkpoint_path = "training_bayes/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every 20 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 verbose=1,
                                                 save_weights_only=True,
                                                 period=20)

log_path = "logs_bayes/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.dirname(log_path)
tensorboard_callback = tf.python.keras.callbacks.TensorBoard(log_dir=logdir)


def make_iterator(dataset):
    iterator = dataset.make_one_shot_iterator()
    next_val = iterator.get_next()

    with K.get_session().as_default() as sess:
        while True:
            pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
            heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl = sess.run(next_val)

            x_trains = {"frustum_point_cloud": pointclouds_pl, "one_hot_vec": one_hot_vec_pl}
            y_trains = {"InsSeg_out": labels_pl, "center_out": centers_pl, "heading_scores": heading_class_label_pl,
                        "size_scores": size_class_label_pl}

            yield x_trains, y_trains


itr_train = make_iterator(parsed_train_dataset)
itr_valid = make_iterator(parsed_val_dataset)
# model_json = model.to_json()
# with open("./training_bayes/model.json", "w") as json_file:
#     print('Saved_model_architecture')
#     json_file.write(model_json)
# model.save('./training_bayes/Bayes_model', save_format='h5')
model.save_weights(checkpoint_path.format(epoch=0))
model.fit_generator(generator=itr_train, validation_data=itr_valid, validation_steps=487, epochs=200,
                    steps_per_epoch=2296, callbacks=[cp_callback, tensorboard_callback], verbose=1, workers=0)

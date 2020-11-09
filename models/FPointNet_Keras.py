# Work adapted to work for Lyft dataset from the original implementation of Frustum PoinNet by Qi et al.
# link to original implementation github repo: https://github.com/charlesq34/frustum-pointnets

from __future__ import print_function

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'train'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import tensorflow.python.keras.backend as K
from tensorflow.python.keras import layers, Model
from tensorflow.python.keras.layers import Lambda

import tensorflow as tf
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from model_util import point_cloud_masking, FPointNet_loss
from model_util import parse_output_to_tensors


def conv_bn(x, filters, trainable, activation='relu'):
    x = layers.Conv2D(filters, kernel_size=[1, 1], strides=[1, 1], padding="valid")(x)
    # x = layers.BatchNormalization()(x)
    if activation == 'relu':
        x = layers.Activation("relu")(x)
    return x


def dense_bn(x, filters, trainable):
    x = layers.Dense(filters)(x)
    # x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)


def PointNetInstanceSeg(point_cloud, one_hot_vec, trainable=False):
    bs = point_cloud.get_shape().as_list()[0]
    n_pts = point_cloud.get_shape().as_list()[1]
    # bs = 32
    # n_pts = 1024
    net = tf.expand_dims(point_cloud, 2)
    net = conv_bn(net, 64, trainable)
    net = conv_bn(net, 64, trainable)
    point_feat = conv_bn(net, 64, trainable)
    net = conv_bn(point_feat, 128, trainable)
    net = conv_bn(net, 1024, trainable)
    global_feat = layers.MaxPool2D(pool_size=(n_pts, 1))(net)

    global_feat = tf.concat([global_feat, tf.expand_dims(tf.expand_dims(one_hot_vec, 1), 1)], axis=3)
    global_feat_expand = tf.tile(global_feat, [1, n_pts, 1, 1])
    concat_feat = tf.concat(axis=3, values=[point_feat, global_feat_expand])

    net = conv_bn(concat_feat, 512, trainable)
    net = conv_bn(net, 256, trainable)
    net = conv_bn(net, 128, trainable)
    net = conv_bn(net, 128, trainable)
    net = layers.Dropout(rate=0.5)(net)
    seg_pred = conv_bn(net, 2, trainable, activation='None')
    seg_pred = tf.squeeze(seg_pred, [2])  # BxNxC

    return seg_pred


def BoxNet(point_cloud, one_hot_vec, trainable=False):
    num_point = point_cloud.get_shape()[1].value
    net = tf.expand_dims(point_cloud, 2)

    net = conv_bn(net, 128, trainable)
    net = conv_bn(net, 128, trainable)
    net = conv_bn(net, 256, trainable)
    net = conv_bn(net, 512, trainable)

    net = layers.MaxPool2D([num_point, 1])(net)
    net = tf.squeeze(net, axis=[1, 2])

    net = tf.concat([net, one_hot_vec], axis=1)  # bs,515

    net = dense_bn(net, 512, trainable)
    net = dense_bn(net, 256, trainable)
    box_pred = layers.Dense(3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER * 4)(net)

    return box_pred


def TNet(object_point_cloud, one_hot_vec, trainable=False):
    # bs = point_cloud.size()[0]
    bs = 32
    num_point = object_point_cloud.get_shape()[1].value

    net = tf.expand_dims(object_point_cloud, 2)
    net = conv_bn(net, 128, trainable)
    net = conv_bn(net, 128, trainable)
    net = conv_bn(net, 256, trainable)
    net = layers.MaxPool2D([num_point, 1])(net)

    net = tf.squeeze(net, axis=[1, 2])

    net = layers.Concatenate(axis=1)([net, one_hot_vec])  # bs,259
    net = dense_bn(net, 256, trainable)
    net = dense_bn(net, 128, trainable)
    center_pred = layers.Dense(3)(net)
    return center_pred


def Frustum_Pointnet_Model(point_cloud_shape=(1024, 3), one_hot_vec_shape=(3,), mask_label_shape=(1024,),
                           center_label_shape=(3,), heading_class_label_shape=(), heading_residual_label_shape=(),
                           size_class_label_shape=(), size_residual_label_shape=(3,), batch_size=32):
    """ Frustum_PointNet model created using Keras layers
    Inputs: point_cloud: The frustum point cloud generated,
            one_hot_vec: Vector which represent the class of the object,
            mask_label: The label for first stage of Instance segmentation,
            center_label: Label for the center of the box,
            heading_class_label: Label for the heading class based on Heading bins,
            heading_residual_label: Label for the residual from the centroid of the Frustum PointCloud,
            size_class_label: Label for the size class,
            size_residual_label: Label for the residual from size class
    Outputs: Training model: Model used for training purposes , there is no output from the model we use the loss
                             function and train the model and save the model weights
             Testing model: Model used for testing purposes , the weights from the training model are used for the
                            inference outputs the model gives all the outputs needed """
    end_points = {}
    point_cloud = tf.keras.layers.Input(point_cloud_shape, name="frustum_point_cloud", batch_size=batch_size)
    one_hot_vec = tf.keras.layers.Input(one_hot_vec_shape, name="one_hot_vec", batch_size=batch_size)
    mask_label = tf.keras.layers.Input(mask_label_shape, name="seg_label", batch_size=batch_size)
    center_label = tf.keras.layers.Input(center_label_shape, name="box3d_center", batch_size=batch_size)
    heading_class_label = tf.keras.layers.Input(heading_class_label_shape, name="angle_class", batch_size=batch_size)
    heading_residual_label = tf.keras.layers.Input(heading_residual_label_shape, name="angle_residual",
                                                   batch_size=batch_size)
    size_class_label = tf.keras.layers.Input(size_class_label_shape, name="size_class", batch_size=batch_size)
    size_residual_label = tf.keras.layers.Input(size_residual_label_shape, name="size_residual", batch_size=batch_size)

    logits = PointNetInstanceSeg(point_cloud, one_hot_vec, trainable=False)  # bs,n,2
    end_points['mask_logits'] = logits

    # Mask Point Centroid
    object_point_cloud_xyz, mask_xyz_mean, end_points = point_cloud_masking(point_cloud, logits, end_points)

    # T-Net
    center_delta = TNet(object_point_cloud_xyz, one_hot_vec, trainable=False)  # (32,3)

    stage1_center = center_delta + mask_xyz_mean  # Bx3
    end_points['stage1_center'] = stage1_center
    # Get object point cloud in object coordinate
    object_point_cloud_xyz_new = object_point_cloud_xyz - tf.expand_dims(center_delta, 1)

    # 3D Box Estimation
    box_pred = BoxNet(object_point_cloud_xyz_new, one_hot_vec)  # (32, 59)

    end_points = parse_output_to_tensors(box_pred, end_points)
    end_points['center'] = end_points['center_boxnet'] + stage1_center  # Bx3

    logits = end_points['mask_logits']
    mask = end_points['mask']
    stage1_center = end_points['stage1_center']
    center_boxnet = end_points['center_boxnet']
    heading_scores = end_points['heading_scores']  # BxNUM_HEADING_BIN
    heading_residuals_normalized = end_points['heading_residuals_normalized']
    heading_residuals = end_points['heading_residuals']
    size_scores = end_points['size_scores']
    size_residuals_normalized = end_points['size_residuals_normalized']
    size_residuals = end_points['size_residuals']
    center = end_points['center']

    logits = Lambda(lambda x: x, name="InsSeg_out")(logits)
    box3d_center = Lambda(lambda x: x, name="center_out")(center)
    heading_scores = Lambda(lambda x: x, name="heading_scores")(heading_scores)
    heading_residual = Lambda(lambda x: x, name="heading_residual")(heading_residuals)
    heading_residuals_normalized = Lambda(lambda x: x, name="heading_residual_norm")(heading_residuals_normalized)
    size_scores = Lambda(lambda x: x, name="size_scores")(size_scores)
    size_residual = Lambda(lambda x: x, name="size_residual")(size_residuals)
    size_residuals_normalized = Lambda(lambda x: x, name="size_residual_norm")(size_residuals_normalized)

    loss = layers.Lambda(FPointNet_loss, output_shape=(1,), name='fp_loss',
                         arguments={'corner_loss_weight': 10.0, 'box_loss_weight': 1.0})([mask_label, center_label,
                                                                                          heading_class_label,
                                                                                          heading_residual_label,
                                                                                          size_class_label,
                                                                                          size_residual_label,
                                                                                          end_points])

    training_model = Model([point_cloud, one_hot_vec, mask_label, center_label, heading_class_label,
                            heading_residual_label, size_class_label, size_residual_label], loss,
                           name='f_pointnet_train')
    det_model = Model(inputs=[point_cloud, one_hot_vec],
                      outputs=[logits, box3d_center, heading_scores, heading_residual, size_scores, size_residual],
                      name='f_pointnet_inference')
    # training_model.summary()
    return training_model, det_model


if __name__ == '__main__':

    model = Frustum_Pointnet_Model()

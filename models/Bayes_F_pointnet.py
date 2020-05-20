import tensorflow as tf
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Dense, BatchNormalization, \
    Activation, Dropout
from tensorflow.contrib.layers import batch_norm
from tensorflow.python.keras.models import Model
import tensorflow_probability as tfp

# from tfp_util import normal_prior

from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from model_util import point_cloud_masking, get_center_regression_net
from model_util import placeholder_inputs, parse_output_to_tensors, get_loss


def normal_prior(prior_std):
    """Defines normal distribution prior for Bayesian neural network."""

    def prior_fn(dtype, shape, name, trainable, add_variable_fn):
        tfd = tfp.distributions
        dist = tfd.Normal(loc=tf.zeros(shape, dtype),
                          scale=dtype.as_numpy_dtype((prior_std)))
        batch_ndims = tf.size(input=dist.batch_shape_tensor())
        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)

    return prior_fn


def conv_block(inputs, filters, kernel_size, strides, prob, bn, bn_decay, padding='valid', activation='relu',
               is_training = 'True'):
    if prob:
        net = tfp.layers.Convolution2DFlipout(filters=filters, kernel_size=kernel_size, strides=strides,
                                              padding=padding, kernel_prior_fn=normal_prior)
    else:
        net = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)

    if bn:
        net = batch_norm(momentum=bn_decay, training=is_training)(net)

    if activation != 'none':
        net = Activation(activation)(net)

    return net


def Dense_block(inputs, num_outputs, prob, bn, bn_decay, activation='relu', is_training = 'True'):
    if prob:
        net = tfp.layers.DenseFlipout(units=num_outputs)(inputs)
    else:
        net = Dense(units=num_outputs)(inputs)

    if bn:
        net = batch_norm(momentum=bn_decay, training = is_training)(net)

    if activation is not None:
        net = Activation(activation)(net)

    return net


def get_instance_seg_v1_net(point_cloud, one_hot_vec, is_training, bn_decay, end_points):
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    net = tf.compat.v1.expand_dims(point_cloud, 2)
    kernel_size = (1, 1)
    strides = (1, 1)

    conv_1 = conv_block(net, filters=64, kernel_size=kernel_size, strides=strides, prob=True, bn=True,
                        bn_decay=bn_decay, is_training = is_training)
    conv_2 = conv_block(conv_1, filters=64, kernel_size=kernel_size, strides=strides, prob=True, bn=True,
                        bn_decay=bn_decay, is_training = is_training)
    point_features = conv_block(conv_2, filters=64, kernel_size=kernel_size, strides=strides, prob=True,
                                bn=True, bn_decay=bn_decay, is_training = is_training)
    conv_3 = conv_block(point_features, filters=128, kernel_size=kernel_size, strides=strides, prob=True,
                        bn=True, bn_decay=bn_decay, is_training = is_training)
    conv_4 = conv_block(conv_3, filters=1024, kernel_size=kernel_size, strides=strides, prob=True,
                        bn=True, bn_decay=bn_decay, is_training = is_training)
    global_features = MaxPooling2D(pool_size=(num_point, 1))(conv_4)

    global_feat = tf.compat.v1.concat([global_features, tf.expand_dims(tf.expand_dims(one_hot_vec, 1), 1)], axis=3)
    global_feat_expand = tf.compat.v1.tile(global_feat, [1, num_point, 1, 1])
    concat_feat = tf.compat.v1.concat(axis=3, values=[point_features, global_feat_expand])

    conv_5 = conv_block(concat_feat, filters=512, kernel_size=kernel_size, strides=strides, prob=True,
                        bn=True, bn_decay=bn_decay, is_training = is_training)

    conv_6 = conv_block(conv_5, filters=256, kernel_size=kernel_size, strides=strides, prob=True,
                        bn=True, bn_decay=bn_decay, is_training = is_training)

    conv_7 = conv_block(conv_6, filters=128, kernel_size=kernel_size, strides=strides, prob=True,
                        bn=True, bn_decay=bn_decay, is_training = is_training)

    conv_8 = conv_block(conv_7, filters=128, kernel_size=kernel_size, strides=strides, prob=True,
                        bn=True, bn_decay=bn_decay, is_training = is_training)

    net = Dropout(rate=0.5)(conv_8)

    logits = conv_block(net, filters=2, kernel_size=kernel_size, strides=strides, prob=True, bn=False,
                        bn_decay=bn_decay, is_training = is_training)

    logits = tf.compat.v1.squeeze(logits, [2])  # BxNxC

    return logits, end_points


def get_3d_box_estimation_v1_net(object_point_cloud, one_hot_vec, is_training, bn_decay, end_points):
    num_point = object_point_cloud.get_shape()[1].value
    net = tf.compat.v1.expand_dims(object_point_cloud, 2)

    kernel_size = (1, 1)
    strides = (1, 1)

    conv_1 = conv_block(net, filters=128, kernel_size=kernel_size, strides=strides, prob=True,
                        bn=True, bn_decay=bn_decay, is_training = is_training)

    conv_2 = conv_block(conv_1, filters=128, kernel_size=kernel_size, strides=strides, prob=True,
                        bn=True, bn_decay=bn_decay, is_training = is_training)

    conv_3 = conv_block(conv_2, filters=256, kernel_size=kernel_size, strides=strides, prob=True,
                        bn=True, bn_decay=bn_decay, is_training = is_training)

    conv_4 = conv_block(conv_3, filters=512, kernel_size=kernel_size, strides=strides, prob=True,
                        bn=True, bn_decay=bn_decay, is_training = is_training)

    maxpool_1 = MaxPooling2D(pool_size=(num_point, 1))(conv_4)

    net = tf.compat.v1.squeeze(maxpool_1, axis=[1, 2])
    net = tf.compat.v1.concat([net, one_hot_vec], axis=1)

    dense_1 = Dense_block(net, 512, prob=True, bn=True, bn_decay=bn_decay, is_training = is_training)
    dense_2 = Dense_block(dense_1, 256, prob=True, bn=True, bn_decay=bn_decay, is_training = is_training)
    output = Dense_block(dense_2, 3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER * 4, prob=True,
                          bn=True, bn_decay=bn_decay, activation='none', is_training = is_training)

    return output, end_points


def get_model(point_cloud, one_hot_vec, is_training, bn_decay=None):
    """ Frustum PointNets model. The model predict 3D object masks and
    amodel bounding boxes for objects in frustum point clouds.

    Input:
        point_cloud: TF tensor in shape (B,N,4)
            frustum point clouds with XYZ and intensity in point channels
            XYZs are in frustum coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
        is_training: TF boolean scalar
        bn_decay: TF float scalar
    Output:
        end_points: dict (map from name strings to TF tensors)
    """
    end_points = {}

    # 3D Instance Segmentation PointNet
    logits, end_points = get_instance_seg_v1_net( \
        point_cloud, one_hot_vec,
        is_training, bn_decay, end_points)
    end_points['mask_logits'] = logits

    # Masking
    # select masked points and translate to masked points' centroid
    object_point_cloud_xyz, mask_xyz_mean, end_points = \
        point_cloud_masking(point_cloud, logits, end_points)

    # T-Net and coordinate translation
    center_delta, end_points = get_center_regression_net( \
        object_point_cloud_xyz, one_hot_vec,
        is_training, bn_decay, end_points)
    stage1_center = center_delta + mask_xyz_mean  # Bx3
    end_points['stage1_center'] = stage1_center
    # Get object point cloud in object coordinate
    object_point_cloud_xyz_new = \
        object_point_cloud_xyz - tf.compat.v1.expand_dims(center_delta, 1)

    # Amodel Box Estimation PointNet
    output, end_points = get_3d_box_estimation_v1_net( \
        object_point_cloud_xyz_new, one_hot_vec,
        is_training, bn_decay, end_points)

    # Parse output to 3D box parameters
    end_points = parse_output_to_tensors(output, end_points)
    end_points['center'] = end_points['center_boxnet'] + stage1_center  # Bx3

    return end_points


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024, 4))
        outputs = get_model(inputs, tf.compat.v1.ones((32, 3)), tf.compat.v1.constant(True))
        for key in outputs:
            print((key, outputs[key]))
        loss = get_loss(tf.compat.v1.zeros((32, 1024), dtype=tf.compat.v1.int32),
                        tf.compat.v1.zeros((32, 3)), tf.compat.v1.zeros((32,), dtype=tf.compat.v1.int32),
                        tf.compat.v1.zeros((32,)), tf.compat.v1.zeros((32,), dtype=tf.compat.v1.int32),
                        tf.compat.v1.zeros((32, 3)), outputs)
        print(loss)

# import keras
# import keras.backend as K
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras.layers import Conv2D, Dense, Concatenate, BatchNormalization, Activation, MaxPooling2D
from tensorflow.python.keras.layers import Lambda, Add
from tensorflow.python.keras.models import Model

from models.model_util import NUM_OBJECT_POINT, NUM_HEADING_BIN, NUM_SIZE_CLUSTER
from models.model_util import g_mean_size_arr, fp_loss


def conv_bn_act(inputs, n_filters=64, kernel=(1, 1), padding='VALID', stride=1, activation_fn='relu', bn=True,
                bn_decay=0.5, scope='conv', is_bayes=True):
    with tf.name_scope(scope):
        if is_bayes:
            x = tfp.layers.Convolution2DFlipout(filters=n_filters, kernel_size=kernel, strides=stride, name=scope,
                                                data_format='channels_last')(inputs)
        else:
            x = Conv2D(n_filters, kernel_size=kernel, strides=stride, data_format='channels_last', name=scope)(inputs)
        if bn:
            x = BatchNormalization(momentum=1 - bn_decay, name=scope + '_BN')(x)
        if activation_fn is not 'none':
            x = Activation(activation_fn, name=scope + '_relu')(x)
        return x


def Dense_bn_act(inputs, num_outputs, activation_fn='relu', bn=True, bn_decay=0.5, scope='fc', is_bayes=True):
    with tf.name_scope(scope):
        if is_bayes:
            x = tfp.layers.DenseFlipout(units=num_outputs, name=scope)(inputs)
        else:
            x = Dense(units=num_outputs, name=scope)(inputs)
        if bn:
            x = BatchNormalization(momentum=1 - bn_decay, name=scope + '_BN')(x)
        if activation_fn is not 'none':
            x = Activation(activation_fn)(x)
        return x


class tf_gather_object_pc(tf.keras.layers.Layer):
    def __init__(self, num_points=512):
        self.num_points = num_points
        super(tf_gather_object_pc, self).__init__()

    def mask_to_indices(self, mask):
        indices = np.zeros((mask.shape[0], self.num_points, 2), dtype=np.int32)
        for i in range(mask.shape[0]):
            pos_indices = np.where(mask[i, :] > 0.5)[0]
            # skip cases when pos_indices is empty
            if len(pos_indices) > 0:
                if len(pos_indices) > self.num_points:
                    choice = np.random.choice(len(pos_indices),
                                              self.num_points, replace=False)
                else:
                    choice = np.random.choice(len(pos_indices),
                                              self.num_points - len(pos_indices), replace=True)
                    choice = np.concatenate((np.arange(len(pos_indices)), choice))
                np.random.shuffle(choice)
                indices[i, :, 1] = pos_indices[choice]
            indices[i, :, 0] = i
        return indices

    def call(self, inputs):
        point_cloud = inputs[0]
        mask = inputs[1]
        indices = Lambda(lambda x: tf.py_function(func=self.mask_to_indices, inp=[x], Tout=tf.int32))(mask)
        object_pc = Lambda(lambda x: tf.gather_nd(x, indices=indices))(point_cloud)
        return object_pc


class parse_output_to_tensors(tf.keras.layers.Layer):
    def __init__(self, end_points):
        self.end_points = end_points
        super(parse_output_to_tensors, self).__init__()

    def call(self, output):
        batch_size = 32
        center = Lambda(lambda x: tf.slice(x, start=[0, 0], size=[-1, 3]))(output)
        self.end_points.append(center)  # 4

        heading_scores = Lambda(lambda x: tf.slice(x, start=[0, 3], size=[-1, NUM_HEADING_BIN]))(output)
        heading_residuals_normalized = Lambda(
            lambda x: tf.slice(x, start=[0, 3 + NUM_HEADING_BIN], size=[-1, NUM_HEADING_BIN]))(output)
        self.end_points.append(heading_scores)  # BxNUM_HEADING_BIN #5
        self.end_points.append(heading_residuals_normalized)  # BxNUM_HEADING_BIN (-1 to 1) #6
        self.end_points.append(heading_residuals_normalized * (np.pi / NUM_HEADING_BIN))  # BxNUM_HEADING_BIN #7

        size_scores = Lambda(lambda x: tf.slice(x, start=[0, 3 + NUM_HEADING_BIN * 2], size=[-1, NUM_HEADING_BIN]))(
            output)

        size_residuals_normalized = Lambda(lambda x: tf.slice(x, start=[0, 3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER],
                                                           size=[-1, NUM_HEADING_BIN * 3]))(output)
        size_residuals_normalized = Lambda(lambda x: tf.reshape(x, shape=[batch_size, NUM_SIZE_CLUSTER, 3]))(
            size_residuals_normalized)
        self.end_points.append(size_scores)  # 8
        self.end_points.append(size_residuals_normalized)  # 9
        self.end_points.append(size_residuals_normalized * \
                               Lambda(lambda x: tf.expand_dims(x, axis=0))(
                                   tf.constant(g_mean_size_arr, dtype=tf.float32)))  # 10
        #         self.end_points.append(Lambda(lambda a: a[0]+a[1])(self.end_points[4] + self.end_points[3])) #11

        return self.end_points


def frustum_pointnet(point_cloud_shape=(1024, 4), one_hot_vec_shape=(3,), g_mean_size_tensor_shape=(3, 3),
                     mask_label_shape=(1024,), center_label_shape=(3,), heading_class_label_shape=(),
                     heading_residual_label_shape=(), size_class_label_shape=(), size_residual_label_shape=(3,)):
    is_training = True
    is_bayes = True
    bn_decay = 0.001
    g_mean_size_tensor = tf.keras.layers.Input(g_mean_size_tensor_shape, name="class_mean_size")
    xyz_only = True
    point_cloud = tf.keras.layers.Input(point_cloud_shape, name="point_cloud")
    one_hot_vec = tf.keras.layers.Input(one_hot_vec_shape, name="one_hot_vector")
    mask_label = tf.keras.layers.Input(mask_label_shape, name="mask_label")
    center_label = tf.keras.layers.Input(center_label_shape, name="center_label")
    heading_class_label = tf.keras.layers.Input(heading_class_label_shape, name="heading_class_label")
    heading_residual_label = tf.keras.layers.Input(heading_residual_label_shape, name="heading_residual_label")
    size_class_label = tf.keras.layers.Input(size_class_label_shape, name="size_class_label")
    size_residual_label = tf.keras.layers.Input(size_residual_label_shape, name="size_residual_label")

    num_point = 1024

    net = Lambda(lambda x: tf.expand_dims(x, axis=2))(point_cloud)

    net = conv_bn_act(net, 64, [1, 1], padding='VALID', stride=[1, 1], bn=True, scope='conv1', bn_decay=bn_decay)
    net = conv_bn_act(net, 64, [1, 1], padding='VALID', stride=[1, 1], bn=True, scope='conv2', bn_decay=bn_decay)
    point_feat = conv_bn_act(net, 64, [1, 1], padding='VALID', stride=[1, 1], bn=True, scope='conv3', bn_decay=bn_decay)
    net = conv_bn_act(point_feat, 128, [1, 1], padding='VALID', stride=[1, 1], bn=True, scope='conv4',
                      bn_decay=bn_decay)
    net = conv_bn_act(net, 1024, [1, 1], padding='VALID', stride=[1, 1], bn=True, scope='conv5', bn_decay=bn_decay)
    global_feat = MaxPooling2D([num_point, 1], padding='VALID', name='maxpool')(net)
    mod_one_hot = Lambda(lambda x: tf.expand_dims(x, axis=1))(one_hot_vec)
    mod_one_hot = Lambda(lambda x: tf.expand_dims(x, axis=1))(mod_one_hot)

    global_feat = Concatenate(axis=3)([global_feat, mod_one_hot])
    global_feat_expand = Lambda(lambda x: tf.tile(x, [1, num_point, 1, 1]))(global_feat)
    concat_feat = Concatenate(axis=3)([point_feat, global_feat_expand])

    net = conv_bn_act(concat_feat, 512, [1, 1], padding='VALID', stride=[1, 1], bn=True, scope='conv6',
                      bn_decay=bn_decay)
    net = conv_bn_act(net, 256, [1, 1], padding='VALID', stride=[1, 1], bn=True, scope='conv7', bn_decay=bn_decay)
    net = conv_bn_act(net, 128, [1, 1], padding='VALID', stride=[1, 1], bn=True, scope='conv8', bn_decay=bn_decay)
    net = conv_bn_act(net, 128, [1, 1], padding='VALID', stride=[1, 1], bn=True, scope='conv9', bn_decay=bn_decay)
    #     net = tf_util.dropout(net, is_training, 'dp1', keep_prob=0.5)

    logits = conv_bn_act(net, 2, [1, 1], padding='VALID', stride=[1, 1], activation_fn='none', scope='conv10')
    logits = Lambda(lambda x: tf.squeeze(x, axis=2), name='logits')(logits)

    ########## Instance segmentation model end #########
    #### Values to be used : logits #####

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    mask = Lambda(lambda a: a[0] < a[1])([Lambda(lambda x: tf.slice(x, begin=[0, 0, 0], size=[-1, -1, 1]))(logits),
                                          Lambda(lambda x: tf.slice(x, begin=[0, 0, 1], size=[-1, -1, 1]))(logits)])

    mask = Lambda(lambda x: tf.cast(x, dtype=float))(mask)

    mask_count = Lambda(lambda x: tf.tile(x, [1, 1, 3]))(Lambda(lambda x: tf.reduce_sum(x, axis=1, keepdims=True))(mask))

    point_cloud_xyz = Lambda(lambda x: tf.slice(x, begin=[0, 0, 0], size=[-1, -1, 3]))(point_cloud)
    mask_repeat = Lambda(lambda x: tf.tile(x, [1, 1, 3]))(mask)

    mask_xyz_mean = Lambda(lambda x: tf.reduce_sum(x, axis=1, keepdims=True))(Lambda(lambda a: a[0] * a[1])
                                                                              ([mask_repeat, point_cloud_xyz]))

    mask = Lambda(lambda x: tf.squeeze(x, axis=[2]), name='mask')(mask)
    mask_xyz_mean = Lambda(lambda a: a[0] / a[1])([mask_xyz_mean, Lambda(lambda x: tf.maximum(x, 1))(mask_count)])

    # Translate to masked points' centroid
    point_cloud_xyz_stage1 = Lambda(lambda a: a[0] - a[1])(
        [point_cloud_xyz, Lambda(lambda x: tf.tile(x, [1, num_point, 1]))(mask_xyz_mean)])

    if xyz_only:
        point_cloud_stage1 = point_cloud_xyz_stage1
    else:
        point_cloud_features = Lambda(lambda x: tf.slice(x, start=[0, 0, 3], size=[-1, -1, -1]))(point_cloud)
        point_cloud_stage1 = Concatenate(axis=-1)([point_cloud_xyz_stage1, point_cloud_features])
    num_channels = point_cloud_stage1.get_shape()[2].value

    object_point_cloud = tf_gather_object_pc(NUM_OBJECT_POINT)([point_cloud_stage1, mask])
    object_point_cloud.set_shape([batch_size, NUM_OBJECT_POINT, num_channels])
    mask_xyz_mean = Lambda(lambda x: tf.squeeze(x, axis=1))(mask_xyz_mean)

    ###### Masking Done ######
    ###### Values to be used : object_point_cloud, mask_xyz_mean, mask ######

    num_point = object_point_cloud.get_shape()[1].value

    net = Lambda(lambda x: tf.expand_dims(x, axis=2))(object_point_cloud)
    net = conv_bn_act(net, 128, [1, 1], padding='VALID', stride=1, bn=True, scope='conv-reg1-stage1', bn_decay=bn_decay)
    net = conv_bn_act(net, 128, [1, 1], padding='VALID', stride=1, bn=True, scope='conv-reg2-stage1', bn_decay=bn_decay)
    net = conv_bn_act(net, 256, [1, 1], padding='VALID', stride=1, bn=True, scope='conv-reg3-stage1', bn_decay=bn_decay)
    net = MaxPooling2D([num_point, 1], padding='VALID', name='maxpool-stage1')(net)
    net = Lambda(lambda x: tf.squeeze(net, axis=[1, 2]))(net)
    net = Concatenate(axis=1)([net, one_hot_vec])
    net = Dense_bn_act(net, 256, scope='fc1-stage1', bn=True, bn_decay=bn_decay)
    net = Dense_bn_act(net, 128, scope='fc2-stage1', bn=True, bn_decay=bn_decay)
    center_delta = Dense_bn_act(net, 3, activation_fn='none', scope='fc3-stage1')

    #     ###### T-net Done #######
    #     ###### Values to be used : center_delta ######

    stage1_center = Add(name='stage1_center')([center_delta, mask_xyz_mean])

    object_point_cloud_xyz_new = Lambda(lambda a: a[0] - a[1])(
        [object_point_cloud, Lambda(lambda x: tf.expand_dims(x, axis=1))(center_delta)])

    num_point = object_point_cloud_xyz_new.get_shape()[1].value
    net = Lambda(lambda x: tf.expand_dims(x, axis=2))(object_point_cloud)
    net = conv_bn_act(net, 128, [1, 1], padding='VALID', stride=1, bn=True, scope='conv-reg1', bn_decay=bn_decay)
    net = conv_bn_act(net, 128, [1, 1], padding='VALID', stride=1, bn=True, scope='conv-reg2', bn_decay=bn_decay)
    net = conv_bn_act(net, 256, [1, 1], padding='VALID', stride=1, bn=True, scope='conv-reg3', bn_decay=bn_decay)
    net = conv_bn_act(net, 512, [1, 1], padding='VALID', stride=1, bn=True, scope='conv-reg4', bn_decay=bn_decay)
    net = MaxPooling2D([num_point, 1], padding='VALID', name='maxpool2')(net)

    net = Lambda(lambda x: tf.squeeze(x, axis=[1, 2]))(net)
    net = Concatenate(axis=1)([net, one_hot_vec])
    net = Dense_bn_act(net, 512, scope='fc1', bn=True, bn_decay=bn_decay)
    net = Dense_bn_act(net, 256, scope='fc2', bn=True, bn_decay=bn_decay)
    output = Dense_bn_act(net, 3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER * 4, activation_fn='none', bn=False)

    ###############################################
    ####### Extracting Necessary O/P needed #######
    ###############################################

    boxnet_center = Lambda(lambda x: tf.slice(x, begin=[0, 0], size=[-1, 3]), name='boxnet_center')(output)

    heading_scores = Lambda(lambda x: tf.slice(x, begin=[0, 3], size=[-1, NUM_HEADING_BIN]), name='heading_scores')(
        output)
    heading_residuals_normalized = Lambda(
        lambda x: tf.slice(x, begin=[0, 3 + NUM_HEADING_BIN], size=[-1, NUM_HEADING_BIN]),
        name='heading_residuals_normalized')(output)
    heading_residuals = Lambda(lambda x: x * (np.pi / NUM_HEADING_BIN), name='heading_residuals')(
        heading_residuals_normalized)

    size_scores = Lambda(lambda x: tf.slice(x, begin=[0, 3 + NUM_HEADING_BIN * 2], size=[-1, NUM_SIZE_CLUSTER]),
                         name='size_scores')(output)
    size_residuals_normalized = Lambda(
        lambda x: tf.slice(x, begin=[0, 3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER], size=[-1, NUM_SIZE_CLUSTER * 3]))(
        output)
    size_residuals_normalized = tf.keras.layers.Reshape((NUM_SIZE_CLUSTER, 3), name='size_residuals_normalized') \
        (size_residuals_normalized)

    g_mean_size_tensor_expd = Lambda(lambda x: tf.expand_dims(x, axis=0))(g_mean_size_tensor)
    size_residuals = Lambda(lambda x: x[0] * x[1], name='size_residuals')([size_residuals_normalized,
                                                                           g_mean_size_tensor_expd])

    center = Lambda(lambda x: x[0] + x[1], name='center')([boxnet_center, stage1_center])

    ###########################################################
    ####### Extracting Predictions for loss calculation #######
    ###########################################################

    # mask_pred = mask
    # mask_logits_pred = logits
    # center_pred = center
    # stage1_center_pred = stage1_center
    # boxnet_center_pred = boxnet_center
    #
    # heading_class_pred = heading_scores
    # heading_residuals_normalized_pred = heading_residuals_normalized
    # heading_residuals_pred = heading_residuals
    #
    # size_scores_pred = size_scores
    # size_residuals_normalized_pred = size_residuals_normalized
    # size_residuals_pred = size_residuals
    loss = Lambda(fp_loss, output_shape=(1,), name='fp_loss',
                  arguments={'corner_loss_weight': 10.0, 'box_loss_weight': 1.0})([mask_label, center_label,
                                                                                   heading_class_label,
                                                                                   heading_residual_label,
                                                                                   size_class_label,
                                                                                   size_residual_label, mask, logits,
                                                                                   center, stage1_center, boxnet_center,
                                                                                   heading_scores,
                                                                                   heading_residuals_normalized,
                                                                                   heading_residuals,
                                                                                   size_scores,
                                                                                   size_residuals_normalized,
                                                                                   size_residuals])

    train_model = Model(inputs=[point_cloud, one_hot_vec, g_mean_size_tensor, mask_label, center_label,
                                heading_class_label, heading_residual_label, size_class_label, size_residual_label],
                        loss=loss)

    detection_model = Model(inputs=[point_cloud, one_hot_vec, g_mean_size_tensor, mask_label, center_label,
                                    heading_class_label, heading_residual_label,
                                    size_class_label, size_residual_label],
                            outputs=[mask, logits, center, stage1_center, boxnet_center,
                                     heading_scores, heading_residuals_normalized, heading_residuals,
                                     size_scores, size_residuals_normalized, size_residuals])
    train_model.summary()

    return train_model, detection_model


if __name__ == '__main__':
    model = frustum_pointnet()
    # model = frustum_pointnet(tf.zeros((32, 1024, 4)), tf.ones((32, 9)), tf.Variable(g_mean_size_arr))
    # model = frustum_pointnet(tf.zeros((32, 1024, 4)), tf.ones((32, 9)), g_mean_size_arr)

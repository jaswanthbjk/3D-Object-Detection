import pickle
import tensorflow as tf
import numpy as np

NUM_HEADING_BIN = 12
NUM_OBJECT_POINT = 512
NUM_SIZE_CLUSTER = 8

g_type2class = {'car': 0, 'Van': 1, 'Truck': 2, 'pedestrian': 3,
                'Person_sitting': 4, 'bicycle': 5, 'Tram': 6, 'Misc': 7}
g_class2type = {g_type2class[t]: t for t in g_type2class}
g_type2onehotclass = {'car': 0, 'pedestrian': 1, 'bicycle': 2}
g_type_mean_size = {'car': np.array([4.76, 1.93, 1.72]),
                    'Van': np.array([5.06763659, 1.9007158, 2.20532825]),
                    'Truck': np.array([10.13586957, 2.58549199, 3.2520595]),
                    'pedestrian': np.array([0.81, 0.77, 1.78]),
                    'Person_sitting': np.array([0.80057803, 0.5983815, 1.27450867]),
                    'bicycle': np.array([1.76, 0.63, 1.44]),
                    'Tram': np.array([16.17150617, 2.53246914, 3.53079012]),
                    'Misc': np.array([3.64300781, 1.54298177, 1.92320313])}
g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))  # size clustrs

for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i, :] = g_type_mean_size[g_class2type[i]]


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# def bytes_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc


def angle2class(angle, num_class):
    ''' Convert continuous angle to discrete class and residual.

    Input:
        angle: rad scalar, from 0-2pi (or -pi~pi), class center at
            0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        num_class: int scalar, number of classes N
    Output:
        class_id, int, among 0,1,...,N-1
        residual_angle: float, a number such that
            class*(2pi/N) + residual_angle = angle
    '''
    angle = angle % (2 * np.pi)
    assert (0 <= angle <= 2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_class)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - \
                     (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle


def class2angle(pred_cls, residual, num_class, to_label_format=True):
    ''' Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
    '''
    angle_per_class = 2 * np.pi / float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle


def size2class(size, type_name):
    """ Convert 3D bounding box size to template class and residuals.
    todo (rqi): support multiple size clusters per type.

    Input:
        size: numpy array of shape (3,) for (l,w,h)
        type_name: string
    Output:
        size_class: int scalar
        size_residual: numpy array of shape (3,)
    """
    size_class = g_type2class[type_name]
    size_residual = size - g_type_mean_size[type_name]
    return size_class, size_residual


def class2size(pred_cls, residual):
    ''' Inverse function to size2class. '''
    mean_size = g_type_mean_size[g_class2type[pred_cls]]
    return mean_size + residual


class tfrecGen_test(object):
    """Convert the pickle file generated during frustum extraction into tensorflow recond file
       inputs:
       pickle_path: Path to the test pickle file
       tfrec_path: Path to save the tensorflow record file"""

    def __init__(self, pickle_path):
        self.npoints = 1024
        self.random_flip = True
        self.random_shift = True
        self.rotate_to_center = True
        self.one_hot = True
        self.from_rgb_detection = False
        self.g_type2onehotclass = {'car': 0, 'pedestrian': 1, 'bicycle': 2}
        with open(pickle_path, 'rb') as fp:
            self.id_list = pickle.load(fp, encoding='latin1')
            self.box2d_list = pickle.load(fp, encoding='latin1')
            self.input_list = pickle.load(fp, encoding='latin1')
            self.type_list = pickle.load(fp, encoding='latin1')
            # frustum_angle is clockwise angle from positive x-axis
            self.frustum_angle_list = pickle.load(fp, encoding='latin1')
            self.prob_list = pickle.load(fp, encoding='latin1')

    def get_center_view_rot_angle(self, index):
        """ Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle """
        return np.pi / 2.0 + self.frustum_angle_list[index]

    def get_center_view_point_set(self, index):
        """ Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        """
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index])
        return rotate_pc_along_y(point_set, self.get_center_view_rot_angle(index))

    def feature_extraction(self, index):
        """ Get index-th element from the picked file dataset. """
        # ------------------------------ INPUTS ----------------------------
        self.class_list = []
        for i in range(len(self.type_list)):
            self.class_list.append(self.g_type2onehotclass[self.type_list[i]])
        rot_angle = self.get_center_view_rot_angle(index)
        scene_id = self.id_list[index]
        box_2D = self.box2d_list[index]
        cls_index = self.class_list[index]
        print(cls_index)

        # Compute one hot vector
        if self.one_hot:
            cls_type = self.type_list[index]
            assert (cls_type in ['car', 'pedestrian', 'bicycle'])
            one_hot_vec = np.zeros(3, dtype=int)
            one_hot_vec[self.g_type2onehotclass[cls_type]] = 1

        if self.rotate_to_center:
            point_set = self.get_center_view_point_set(index)
        else:
            point_set = self.input_list[index]

        # Resample
        if point_set.shape[0] < self.npoints:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        else:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)

        point_set = point_set[choice, :3]

        if self.one_hot:
            return point_set, rot_angle, self.prob_list[index], one_hot_vec, cls_index, scene_id, box_2D
        else:
            return point_set, rot_angle, self.prob_list[index]

    def serialize_example(self, point_set, rot_angle, prob_value, one_hot_vec, cls_index, scene_id, box_2D):
        feature = {'frustum_point_cloud': float_list_feature(point_set.ravel()),
                   'rot_angle': float_feature(rot_angle),
                   'one_hot_vec': int64_list_feature(one_hot_vec),
                   'prob': float_feature(prob_value),
                   'type_name': int64_feature(cls_index),
                   'sample_token': bytes_feature(scene_id.encode('utf-8')),
                   'box_2d': float_list_feature(box_2D.ravel())}
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto

    def write_tfrec(self, tfrec_name):
        with tf.io.TFRecordWriter(tfrec_name) as tfrw:
            for i in range(len(self.id_list)):
                point_set, rot_angle, prob, one_hot_vec, cls_index, scene_id, box_2D = self.feature_extraction(i)
                tfexample = self.serialize_example(point_set, rot_angle, prob, one_hot_vec, cls_index, scene_id, box_2D)
                tfrw.write(tfexample.SerializeToString())


class tfrec_Gen_Train_Val(object):
    def __init__(self, pickle_path):
        self.npoints = 1024
        self.random_flip = True
        self.random_shift = True
        self.rotate_to_center = True
        self.one_hot = True
        self.from_rgb_detection = False
        with open(pickle_path, 'rb') as fp:
            self.id_list = pickle.load(fp, encoding='latin1')
            self.box2d_list = pickle.load(fp, encoding='latin1')
            self.box3d_list = pickle.load(fp, encoding='latin1')
            self.input_list = pickle.load(fp, encoding='latin1')
            self.label_list = pickle.load(fp, encoding='latin1')
            self.type_list = pickle.load(fp, encoding='latin1')
            self.heading_list = pickle.load(fp, encoding='latin1')
            self.size_list = pickle.load(fp, encoding='latin1')
            # frustum_angle is clockwise angle from positive x-axis
            self.frustum_angle_list = pickle.load(fp, encoding='latin1')
            print(len(self.id_list))

    def get_box3d_center(self, index):
        """ Get the center (XYZ) of 3D bounding box. """
        box3d_center = (self.box3d_list[index][0, :] + self.box3d_list[index][6, :]) / 2.0
        return box3d_center

    def get_center_view_box3d_center(self, index):
        """ Frustum rotation of 3D bounding box center. """
        box3d_center = (self.box3d_list[index][0, :] +
                        self.box3d_list[index][6, :]) / 2.0
        return rotate_pc_along_y(np.expand_dims(box3d_center, 0),
                                 self.get_center_view_rot_angle(index)).squeeze()

    def get_center_view_box3d(self, index):
        """ Frustum rotation of 3D bounding box corners. """
        box3d = self.box3d_list[index]
        box3d_center_view = np.copy(box3d)
        return rotate_pc_along_y(box3d_center_view, self.get_center_view_rot_angle(index))

    def get_center_view_point_set(self, index):
        """ Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        """
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index])
        return rotate_pc_along_y(point_set, self.get_center_view_rot_angle(index))

    def get_center_view_rot_angle(self, index):
        """ Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle """
        return np.pi / 2.0 + self.frustum_angle_list[index]

    def feature_extraction(self, index):
        """ Get index-th element from the picked file dataset. """
        # ------------------------------ INPUTS ----------------------------
        rot_angle = self.get_center_view_rot_angle(index)

        # Compute one hot vector
        if self.one_hot:
            cls_type = self.type_list[index]
            assert (cls_type in ['car', 'pedestrian', 'bicycle'])
            one_hot_vec = np.zeros(3, dtype=int)
            one_hot_vec[g_type2onehotclass[cls_type]] = 1

        # Get point cloud
        if self.rotate_to_center:
            point_set = self.get_center_view_point_set(index)
        else:
            point_set = self.input_list[index]

        # Resample
        if point_set.shape[0] < self.npoints:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        else:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)

        point_set = point_set[choice, :3]

        # ------------------------------ LABELS ----------------------------
        seg = self.label_list[index]
        seg = seg[choice]

        # Get center point of 3D box
        if self.rotate_to_center:
            box3d_center = self.get_center_view_box3d_center(index)
        else:
            box3d_center = self.get_box3d_center(index)

        # Heading
        if self.rotate_to_center:
            heading_angle = self.heading_list[index] - rot_angle
        else:
            heading_angle = self.heading_list[index]

        # Size
        size_class, size_residual = size2class(self.size_list[index], self.type_list[index])

        # Data Augmentation
        if self.random_flip:
            # note: rot_angle won't be correct if we have random_flip
            # so do not use it in case of random flipping.
            if np.random.random() > 0.5:  # 50% chance flipping
                point_set[:, 0] *= -1
                box3d_center[0] *= -1
                heading_angle = np.pi - heading_angle
        if self.random_shift:
            dist = np.sqrt(np.sum(box3d_center[0] ** 2 + box3d_center[1] ** 2))
            shift = np.clip(np.random.randn() * dist * 0.05, dist * 0.8, dist * 1.2)
            point_set[:, 2] += shift
            box3d_center[2] += shift

        angle_class, angle_residual = angle2class(heading_angle,
                                                  NUM_HEADING_BIN)

        if self.one_hot:
            return point_set, seg, box3d_center, angle_class, angle_residual, \
                   size_class, size_residual, rot_angle, one_hot_vec
        else:
            return point_set, seg, box3d_center, angle_class, angle_residual, \
                   size_class, size_residual, rot_angle

    def serialize_example(self, point_set, seg, box3d_center, angle_class, angle_residual, size_class, size_residual,
                          rot_angle, one_hot_vec):
        feature = {'frustum_point_cloud': float_list_feature(point_set.ravel()),
                   'seg_label': float_list_feature(seg),
                   'box3d_center': float_list_feature(box3d_center),
                   'angle_class': int64_feature(angle_class),
                   'angle_residual': float_feature(angle_residual),
                   'size_class': int64_feature(size_class),
                   'size_residual': float_list_feature(size_residual.ravel()),
                   'rot_angle': float_feature(rot_angle),
                   'one_hot_vec': float_list_feature(one_hot_vec)}
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto

    def write_tfrec(self, tfrec_name):
        with tf.io.TFRecordWriter(tfrec_name) as tfrw:
            for i in range(len(self.id_list)):
                point_set, seg, box3d_center, angle_class, angle_residual, \
                size_class, size_residual, rot_angle, one_hot_vec = self.feature_extraction(i)
                tfexample = self.serialize_example(point_set, seg, box3d_center, angle_class, angle_residual, size_class,
                                              size_residual,
                                              rot_angle, one_hot_vec)
                tfrw.write(tfexample.SerializeToString())

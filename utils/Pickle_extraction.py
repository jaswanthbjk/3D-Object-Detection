from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
import argparse

from absl import logging
from tqdm import tqdm

# Dataset helpers and loading utils -----------------------------------------------------------------


from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.geometry_utils import BoxVisibility, box_in_image, view_points, points_in_box, \
    transform_matrix
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion

from pyquaternion import Quaternion

# data_path, json_path = r'F:\\LyftDataset\\v1.01-train', r'F:\\LyftDataset\\v1.01-train\\v1.01-train'
data_path, json_path = r'F:\\v1_02\\v1.02-train', r'F:\\v1_02\\v1.02-train\\v1.02-train'
object_of_interest_type = ['animal', 'bicycle', 'bus', 'car', 'emergency_vehicle', 'motorcycle',
                           'other_vehicle', 'pedestrian', 'truck']

g_type2onehotclass = {'animal': 0, 'bicycle': 1, 'bus': 2, 'car': 3, 'emergency_vehicle': 4, 'motorcycle': 5,
                      'other_vehicle': 6, 'pedestrian': 7, 'truck': 8}
g_type_mean_size = {'animal': np.array([0.704, 0.313, 0.489]),
                    'bicycle': np.array([1.775, 0.654, 1.276]),
                    'bus': np.array([11.784, 2.956, 3.416]),
                    'car': np.array([4.682, 1.898, 1.668]),
                    'emergency_vehicle': np.array([5.357, 2.028, 1.852]),
                    'motorcycle': np.array([2.354, 0.942, 1.163]),
                    'other_vehicle': np.array([8.307, 2.799, 3.277]),
                    'pedestrian': np.array([0.787, 0.768, 1.79]),
                    'truck': np.array([8.784, 2.866, 3.438])}


def load_data(data_path, json_path):
    # data_path, json_path = r'F:\\Lyft_Level5_Dataset', r'F:\\Lyft_Level5_Dataset\\train_data'

    dataset_pickle = 'lyft_dataset.pickle'
    if os.path.exists(os.path.join(data_path, dataset_pickle)):
        with open(os.path.join(data_path, dataset_pickle), 'rb') as fp:
            level5data = pickle.load(fp)
    else:
        level5data = LyftDataset(data_path, json_path, verbose=True)
        with open(os.path.join(data_path, dataset_pickle), 'wb') as fp:
            pickle.dump(level5data, fp)

    return level5data


LyftData = load_data(data_path, json_path)


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def size2class(size, obj_type):
    size_class = g_type2onehotclass[obj_type]
    size_residual = size - g_type_mean_size[obj_type]
    return size_class, size_residual


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def map_pointcloud_to_image(pointsensor_token: str, camera_token: str):
    """
    Adapted from lyft_dataset_sdk
    """

    cam = LyftData.get("sample_data", camera_token)
    pointsensor = LyftData.get("sample_data", pointsensor_token)
    pcl_path = LyftData.data_path / pointsensor["filename"]
    pc3d = LidarPointCloud.from_file(pcl_path)
    image = Image.open(str(LyftData.data_path / cam["filename"]))

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = LyftData.get("calibrated_sensor", pointsensor["calibrated_sensor_token"])
    pc3d.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
    pc3d.translate(np.array(cs_record["translation"]))

    # Second step: transform to the global frame.
    poserecord = LyftData.get("ego_pose", pointsensor["ego_pose_token"])
    pc3d.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
    pc3d.translate(np.array(poserecord["translation"]))

    # Third step: transform into the ego vehicle frame for the timestamp of the image.
    poserecord = LyftData.get("ego_pose", cam["ego_pose_token"])
    pc3d.translate(-np.array(poserecord["translation"]))
    pc3d.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

    # Fourth step: transform into the camera.
    cs_record = LyftData.get("calibrated_sensor", cam["calibrated_sensor_token"])
    pc3d.translate(-np.array(cs_record["translation"]))
    pc3d.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

    pc2d = view_points(pc3d.points[:3, :], np.array(cs_record["camera_intrinsic"]), normalize=True)

    return pc3d, pc2d


def extract_pc_in_box2d(pc, box2d):
    """ pc: (N,2), box2d: (xmin,ymin,xmax,ymax) """
    box2d_corners = np.zeros((4, 2))
    box2d_corners[0, :] = [box2d[0], box2d[1]]
    box2d_corners[1, :] = [box2d[2], box2d[1]]
    box2d_corners[2, :] = [box2d[2], box2d[3]]
    box2d_corners[3, :] = [box2d[0], box2d[3]]
    box2d_roi_inds = in_hull(pc[:, 0:2], box2d)
    return pc[box2d_roi_inds, :], box2d_roi_inds


def extract_pc_in_box3d(input_pc, input_box3d):
    assert input_box3d.shape == (3, 8)
    assert input_pc.shape[0] == 3
    pc = np.transpose(input_pc)
    box3d = np.transpose(input_box3d)

    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds.astype(int)


def mask_points(points: np.ndarray, xmin, xmax, ymin, ymax, depth_min=0, buffer_pixel=1) -> np.ndarray:
    depths = points[2, :]

    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > depth_min)
    mask = np.logical_and(mask, points[0, :] > xmin + buffer_pixel)
    mask = np.logical_and(mask, points[0, :] < xmax - buffer_pixel)
    mask = np.logical_and(mask, points[1, :] > ymin + buffer_pixel)
    mask = np.logical_and(mask, points[1, :] < ymax - buffer_pixel)

    return mask


def get_box_corners(transformed_box: Box,
                    cam_intrinsic_mtx: np.array,
                    frustum_pointnet_convention=True):
    box_corners_on_cam_coord = transformed_box.corners()

    if frustum_pointnet_convention:
        rearranged_idx = [0, 3, 7, 4, 1, 2, 6, 5]
        box_corners_on_cam_coord = box_corners_on_cam_coord[:, rearranged_idx]

        assert np.allclose((box_corners_on_cam_coord[:, 0] + box_corners_on_cam_coord[:, 6]) / 2,
                           np.array(transformed_box.center))

    # For perspective transformation, the normalization should set to be True
    box_corners_on_image = view_points(box_corners_on_cam_coord, view=cam_intrinsic_mtx, normalize=True)

    return box_corners_on_image


def get_2d_obj_corners(box_3d_corners_cam_frame):
    assert box_3d_corners_cam_frame.shape[0] == 3

    xmin = box_3d_corners_cam_frame[0, :].min()
    xmax = box_3d_corners_cam_frame[0, :].max()
    ymin = box_3d_corners_cam_frame[1, :].min()
    ymax = box_3d_corners_cam_frame[1, :].max()

    return [xmin, xmax, ymin, ymax]


def get_heading_angle(box: Box):
    box_corners = box.corners()
    v = box_corners[:, 0] - box_corners[:, 4]
    heading_angle = np.arctan2(-v[2], v[0])
    return heading_angle


def transform_image_to_cam_coordinate(image_array_p: np.array, camera_token: str, lyftd: LyftDataset):
    sd_record = lyftd.get("sample_data", camera_token)
    cs_record = lyftd.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = lyftd.get("sensor", cs_record["sensor_token"])
    pose_record = lyftd.get("ego_pose", sd_record["ego_pose_token"])

    # inverse the viewpoint transformation
    def normalization(input_array):
        input_array[0:2, :] = input_array[0:2, :] * input_array[2:3, :].repeat(2, 0).reshape(2, input_array.shape[1])
        return input_array

    image_array = normalization(np.copy(image_array_p))
    image_array = np.concatenate((image_array.ravel(), np.array([1])))
    image_array = image_array.reshape(4, 1)

    cam_intrinsic_mtx = np.array(cs_record["camera_intrinsic"])
    view = cam_intrinsic_mtx
    viewpad = np.eye(4)
    viewpad[: view.shape[0], : view.shape[1]] = view
    image_in_cam_coord = np.dot(np.linalg.inv(viewpad), image_array)

    return image_in_cam_coord[0:3, :]


def get_frustum_angle(dataset, cam_token, box2D):
    random_depth = 20
    xmin, xmax, ymin, ymax = box2D[0], box2D[1], box2D[2], box2D[3]
    image_center = np.array([[(xmax + xmin) / 2, (ymax + ymin) / 2, random_depth]]).T
    image_center_in_cam_coord = transform_image_to_cam_coordinate(image_center, cam_token, dataset)
    assert image_center_in_cam_coord.shape[1] == 1
    frustum_angle = -np.arctan2(image_center_in_cam_coord[2, 0], image_center_in_cam_coord[0, 0])
    return frustum_angle


def rot_y(points, rot_angle):
    rot_mat = np.array([[np.cos(rot_angle), -np.sin(rot_angle)], [np.sin(rot_angle), np.cos(rot_angle)]])
    points[:, [0, 2]] = np.dot(points[:, [0, 2]], np.transpose(rot_mat))
    return points


def angle2class(angle, num_class):
    angle = angle % (2 * np.pi)
    assert (0 <= angle <= 2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_class)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle


def get_angle_class_residual(rotated_heading_angle):
    angle_class, angle_residual = angle2class(rotated_heading_angle, 12)

    return angle_class, angle_residual


class frustum_pc_extractor():
    def __init__(self, lyftd: LyftData, Bbox: Box, pc_in_box, box_3d, box_2d, heading_angle, frustum_angle,
                 sample_token, camera_token, seg_label):
        self.dataset = lyftd
        self.Bbox = Bbox
        self.box_3d = box_3d
        self.box_2d = box_2d
        self.heading_angle = heading_angle
        self.frustum_angle = frustum_angle
        self.sample_token = sample_token
        self.camera_token = camera_token

        self.NUM_POINT = 1024
        sel_index = np.random.choice(pc_in_box.shape[0], self.NUM_POINT)
        self.pc_in_box = pc_in_box[sel_index, :]  # Nx3

        self.seg_label = seg_label[sel_index]
        self.box3d_center = np.copy(self.Bbox.center)
        self.object_name = self.Bbox.name

    def calc_rotation_angle_center(self):
        return np.pi / 2.0 + self.frustum_angle

    def perform_rotation_to_center(self):
        box3d_center = np.copy(self.Bbox.center)
        return rot_y(np.expand_dims(box3d_center, 0),
                     rot_angle=self.calc_rotation_angle_center()).squeeze()

    def get_size_class_residual(self):
        # TODO size2class() and settings were copied from size, we therefore use
        # self._get_wlh() instead of self.box_sensor_coord.size
        size_class, size_residual = size2class(self.get_wlh(), self.Bbox.name)
        return size_class, size_residual

    def get_one_hot_vec(self):
        one_hot_vec = np.zeros(len(g_type2onehotclass), dtype=np.int)
        one_hot_vec[g_type2onehotclass[self.object_name]] = 1
        return one_hot_vec

    def get_rotated_heading_angle(self):
        return self.heading_angle - self.frustum_angle

    def get_camera_intrinsic(self) -> np.ndarray:
        sd_record = self.dataset.get("sample_data", self.camera_token)
        cs_record = self.dataset.get("calibrated_sensor", sd_record["calibrated_sensor_token"])

        camera_intrinsic = np.array(cs_record['camera_intrinsic'])

        return camera_intrinsic

    def get_wlh(self):
        w, l, h = self.Bbox.wlh
        size_lwh = np.array([l, w, h])
        return size_lwh

    def flat_pointcloud(self):
        # not support lidar data with intensity yet
        assert self.pc_in_box.shape[1] == 3

        return self.pc_in_box.ravel()

    def to_train_example(self) -> tf.train.Example:
        rotated_heading_angle = self.heading_angle - self.frustum_angle
        rotated_angle_class, rotated_angle_residual = get_angle_class_residual(rotated_heading_angle)

        size_class, size_residual = self.get_size_class_residual()

        feature_dict = {
            'box3d_size': float_list_feature(self.get_wlh()),  # (3,)
            'size_class': int64_feature(size_class),
            'size_residual': float_list_feature(size_residual.ravel()),  # (3,)

            'frustum_point_cloud': float_list_feature(self.flat_pointcloud()),  # (N,3)
            'rot_frustum_point_cloud': float_list_feature(
                rot_y(self.pc_in_box, rot_angle=np.pi / 2.0 + self.frustum_angle).ravel()),  # (N,3)

            'seg_label': int64_list_feature(self.seg_label.ravel()),

            'box_3d': float_list_feature(self.box_3d.ravel()),  # (8,3)
            'rot_box_3d': float_list_feature(rot_y(self.box_3d, rot_angle=(np.pi / 2.0 + self.frustum_angle)).ravel()),
            # (8,3)

            'box_2d': float_list_feature(self.box_2d.ravel()),  # (4,)

            'heading_angle': float_feature(self.heading_angle),
            'rot_heading_angle': float_feature(self.get_rotated_heading_angle()),
            'rot_angle_class': int64_feature(rotated_angle_class),
            'rot_angle_residual': float_feature(rotated_angle_residual),

            'frustum_angle': float_feature(self.frustum_angle),
            'sample_token': bytes_feature(self.sample_token.encode('utf8')),
            'type_name': bytes_feature(self.Bbox.name.encode('utf8')),
            'one_hot_vec': int64_list_feature(self.get_one_hot_vec()),

            'camera_token': bytes_feature(self.camera_token.encode('utf8')),
            'annotation_token': bytes_feature(self.Bbox.token.encode('utf8')),

            'box_center': float_list_feature(self.Bbox.center.ravel()),  # (3,)
            'rot_box_center': float_list_feature(rot_y(np.expand_dims(self.box3d_center, 0),
                                                       rot_angle=(np.pi / 2.0 + self.frustum_angle)).ravel()),  # (3,)

        }
        sample = tf.train.Example(features=tf.train.Features(feature=feature_dict))

        return sample


class FrustumExtractor(object):

    def __init__(self, sample_token: str, lyftd: LyftData, camera_type=None, use_multisweep=False):
        self.object_of_interest_type = object_of_interest_type

        if camera_type is None:
            camera_type = ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT',
                           'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']
        self.dataset = lyftd
        self.sample_record = self.dataset.get("sample", sample_token)
        self.camera_type = camera_type
        self.camera_keys = self.extract_camera_keys()
        # self.pcl_data, self.pcl_token = self.read_pcl_data(use_multisweep)
        try:
            self.pcl_data, self.pcl_token = self.read_pcl_data(use_multisweep=False)
        except Exception:
            pass
        self.pcl_in_cam_frame = {}
        self.multi_sweep = use_multisweep

    def extract_camera_keys(self):
        cams = [key for key in self.sample_record["data"].keys() if "CAM" in key]
        cams = [cam for cam in cams if cam in self.camera_type]
        return cams

    def read_pcl_data(self, use_multisweep=False):
        pcl_token = self.sample_record['data']['LIDAR_TOP']
        pcl_path = self.dataset.get_sample_data_path(pcl_token)
        print(pcl_path)
        if use_multisweep:
            pc, _ = LidarPointCloud.from_file_multisweep(self.dataset, self.sample_record, chan='LIDAR_TOP',
                                                         ref_chan='LIDAR_TOP', num_sweeps=26)
        else:

            pc = LidarPointCloud.from_file(pcl_path)

        return pc, pcl_token

    def read_image_paths(self):
        for cam in self.camera_keys:
            cam_token = self.sample_record['data'][cam]
            img_path = self.dataset.get_sample_data_path(cam_token)
            yield img_path

    def Frustums_from_gt(self):
        clip_distance = 2.0
        max_clip_distance = 60
        for cam_key in self.camera_keys:
            cam_token = self.sample_record['data'][cam_key]
            cam_data = self.dataset.get('sample_data', cam_token)
            try:
                pc, pc_token = self.read_pcl_data(use_multisweep=False)
            except Exception:
                continue
            image_path, box_list, cam_intrinsic = self.dataset.get_sample_data(cam_token,
                                                                               box_vis_level=BoxVisibility.ANY,
                                                                               selected_anntokens=None)
            pc_3d, pc_2d = map_pointcloud_to_image(pc_token, cam_token)
            self.pcl_in_cam_frame[cam_token] = pc_3d
            img = Image.open(image_path)

            for box in box_list:
                mask = mask_points(pc_2d, 0, img.size[0], ymin=0, ymax=img.size[1])
                distance_mask = (pc.points[2, :] > clip_distance) & (pc.points[2, :] < max_clip_distance)
                mask = np.logical_and(mask, distance_mask)
                box_in_lens_frame = get_box_corners(box, cam_intrinsic, frustum_pointnet_convention=True)
                obj_2d_corners = get_2d_obj_corners(box_in_lens_frame)
                xmin, xmax, ymin, ymax = obj_2d_corners[0], obj_2d_corners[1], obj_2d_corners[2], obj_2d_corners[3]
                box_mask = mask_points(pc_2d, xmin, xmax, ymin, ymax)
                mask = np.logical_and(mask, box_mask)
                point_clouds_in_box = pc.points[:, mask]
                pc_in_box_roi, seg_label = extract_pc_in_box3d(point_clouds_in_box[0:3, :], box.corners())
                heading_angle = get_heading_angle(box)
                frustum_angle = get_frustum_angle(self.dataset, cam_token, obj_2d_corners)
                box_2D = np.array(obj_2d_corners)
                box_3D = np.transpose(box.corners())
                point_clouds_in_box = point_clouds_in_box[0:3, :]
                point_clouds_in_box = np.transpose(point_clouds_in_box)
                if box.name not in object_of_interest_type:
                    continue
                if point_clouds_in_box.shape[0] < 300:
                    continue

                fp = frustum_pc_extractor(lyftd=self.dataset, Bbox=box,
                                          pc_in_box=point_clouds_in_box,
                                          box_3d=box_3D, box_2d=box_2D, heading_angle=heading_angle,
                                          frustum_angle=frustum_angle, camera_token=cam_token,
                                          sample_token=self.sample_record['token'], seg_label=seg_label)

                yield fp


def get_all_boxes_in_single_scene(scene_number, from_rgb_detection, ldf, use_multisweep=False,
                                  object_classifier=None):
    start_sample_token = ldf.scene[scene_number]['first_sample_token']
    sample_token = start_sample_token
    counter = 0
    while sample_token != "":
        if counter % 10 == 0:
            logging.info("Processing {} token {}".format(scene_number, counter))
        counter += 1
        sample_record = ldf.get('sample', sample_token)
        fg = FrustumExtractor(sample_token, ldf, use_multisweep=use_multisweep)
        if not from_rgb_detection:
            for fp in fg.Frustums_from_gt():
                yield fp
        # else:
        #     # reserved for rgb detection data
        #     for fp in fg.generate_frustums_from_2d(object_classifier):
        #         yield fp

        next_sample_token = sample_record['next']
        sample_token = next_sample_token


if __name__ == '__main__':
    data_type = "train"
    file_type = "gt"
    scene_list = range(148)
    for i in tqdm(scene_list):
        with tf.io.TFRecordWriter(
                os.path.join('./',
                             "scene_{0}_{1}_{2}.tfrec".format(i, data_type, file_type))) as tfrw:
            for fp in get_all_boxes_in_single_scene(scene_number=i, from_rgb_detection=False,
                                                    ldf=LyftData,
                                                    use_multisweep=False,
                                                    object_classifier=None):
                tfexample = fp.to_train_example()
                tfrw.write(tfexample.SerializeToString())

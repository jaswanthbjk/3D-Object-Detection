from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import pickle
import argparse
import json
from pathlib import Path

# Dataset helpers and loading utils -----------------------------------------------------------------


from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.geometry_utils import BoxVisibility, box_in_image, view_points, points_in_box, \
    transform_matrix
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion

import copy
from pyquaternion import Quaternion

import cv2

data_path, json_path = r'F:\\Lyft_Level5_Dataset', r'F:\\Lyft_Level5_Dataset\\train_data'
object_of_interest_type = ['animal', 'bicycle', 'bus', 'car', 'emergency_vehicle', 'motorcycle',
                             'other_vehicle', 'pedestrian', 'truck']


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

g_type2onehotclass = {'animal': 0, 'bicycle': 1, 'bus': 2, 'car': 3, 'emergency_vehicle': 4, 'motorcycle': 5,
                      'other_vehicle': 6, 'pedestrian': 7, 'truck': 8}


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


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


def read_2d_labels(img_name):
    path = r'F:\v1.01-train\Bbox2D'
    txt_name = img_name[7:-5] + '.txt'
    with open(os.path.join(path, txt_name), "r", encoding="utf-8") as f:
        content = eval(f.read())
    return content


def quaternion_yaw(q: Quaternion, in_image_frame: bool = True) -> float:
    if in_image_frame:
        v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
        yaw = -np.arctan2(v[2], v[0])
    else:
        v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])

    return yaw


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

    box3d_roi_inds = in_hull(pc[:, 0:3], input_box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


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


def get_heading_angle(box):
    box_corners = box.corners
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
    residual_angle = shifted_angle - \ (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle


def get_angle_class_residual(rotated_heading_angle):
    angle_class, angle_residual = angle2class(rotated_heading_angle, 12)

    return angle_class, angle_residual


class Frustum_pc_extractor():
    def __init__(self, lyftd: LyftData, Bbox, pc_in_box, box_3d, box_2d, heading_angle, frustum_angle,
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

    def calc_rotation_angle_center(self):
        return np.pi / 2.0 + self.frustum_angle

    def perform_rotation_to_center(self):
        box3d_center = np.copy(self.Bbox.center)
        return rot_y(np.expand_dims(box3d_center, 0),
                     rot_angle=self.calc_rotation_angle_center()).squeeze()

    def get_size_class_residual(self):
        # TODO size2class() and settings were copied from size, we therefore use
        # self._get_wlh() instead of self.box_sensor_coord.size
        size_class, size_residual = size2class(self._get_wlh(), self.Bbox.name)
        return size_class, size_residual

    def _get_one_hot_vec(self):
        one_hot_vec = np.zeros(len(g_type2onehotclass), dtype=np.int)
        one_hot_vec[g_type2onehotclass[self.object_name]] = 1
        return one_hot_vec

    def _get_rotated_heading_angle(self):
        return self.heading_angle - self.frustum_angle

    def _get_camera_intrinsic(self) -> np.ndarray:
        sd_record = self.dataset.get("sample_data", self.camera_token)
        cs_record = self.dataset.get("calibrated_sensor", sd_record["calibrated_sensor_token"])

        camera_intrinsic = np.array(cs_record['camera_intrinsic'])

        return camera_intrinsic

    def _get_wlh(self):
        w, l, h = self.Bbox.wlh
        size_lwh = np.array([l, w, h])
        return size_lwh

    def _flat_pointcloud(self):
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
            'rot_frustum_point_cloud': float_list_feature(self.get_rotated_point_cloud().ravel()),  # (N,3)

            'seg_label': int64_list_feature(self.seg_label.ravel()),

            'box_3d': float_list_feature(self.box_3d.ravel()),  # (8,3)
            'rot_box_3d': float_list_feature(self.get_rotated_box_3d().ravel()),  # (8,3)

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
            'rot_box_center': float_list_feature(self.get_rotated_center().ravel()),  # (3,)

        }
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

        return example


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
        self.pcl_data, self.pcl_token = self.read_pcl_data(use_multisweep)
        self.pcl_in_cam_frame = {}
        self.multi_sweep = use_multisweep

    def extract_camera_keys(self):
        cams = [key for key in self.sample_record["data"].keys() if "CAM" in key]
        cams = [cam for cam in cams if cam in self.camera_type]
        return cams

    def read_pcl_data(self, use_multisweep=False):
        pcl_token = self.sample_record['data']['LIDAR_TOP']
        pcl_path = self.dataset.get_sample_data_path(pcl_token)
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

            pc, pc_token = self.read_pcl_data(use_multisweep=False)
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
                frustum_points = frustum_pc_extractor()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_train', action='store_true',
                        help='Generate train split frustum data with perturbed GT 2D boxes')

    parser.add_argument('--gen_val', action='store_true', help='Generate val split frustum data with GT 2D boxes')

    parser.add_argument('--gen_val_rgb_detection', action='store_true',
                        help='Generate val split frustum data with RGB detection 2D boxes')

    parser.add_argument('--car_only', action='store_true', help='Only generate cars')
    parser.add_argument('--people_only', action='store_true', help='Only generate peds and cycs')
    parser.add_argument('--save_dir', default=None, type=str, help='data directory to save data')

    args = parser.parse_args()

    np.random.seed(3)

    if args.save_dir is None:
        save_dir = 'kitti/data/pickle_data'
    else:
        save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.car_only:
        type_whitelist = ['Car']
        output_prefix = 'frustum_caronly_'

    elif args.people_only:
        type_whitelist = ['Pedestrian', 'Cyclist']
        output_prefix = 'frustum_pedcyc_'

    else:
        type_whitelist = ['Car', 'Pedestrian', 'Cyclist']
        output_prefix = 'frustum_carpedcyc_'

    if args.gen_train:
        extract_frustum_data(
            os.path.join(BASE_DIR, 'image_sets/train.txt'),
            'training',
            os.path.join(save_dir, output_prefix + 'train.pickle'),
            perturb_box2d=True, augmentX=5,
            type_whitelist=type_whitelist)

    if args.gen_val:
        extract_frustum_data(
            os.path.join(BASE_DIR, 'image_sets/val.txt'),
            'training',
            os.path.join(save_dir, output_prefix + 'val.pickle'),
            perturb_box2d=False, augmentX=1,
            type_whitelist=type_whitelist)

    if args.gen_val_rgb_detection:
        extract_frustum_data_rgb_detection(
            os.path.join(BASE_DIR, 'rgb_detections/rgb_detection_val.txt'),
            'training',
            os.path.join(save_dir, output_prefix + 'val_rgb_detection.pickle'),
            type_whitelist=type_whitelist)

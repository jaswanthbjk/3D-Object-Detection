from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from PIL import Image
import pickle
import argparse

# Dataset helpers and koading utils -----------------------------------------------------------------
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix

level5data = LyftDataset(data_path=r'F:\\LyftDataset\\v1.01-train', json_path=r'F:\\LyftDataset\\v1.01-train\\v1.01-train',
                      verbose=True)

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.geometry_utils import BoxVisibility, box_in_image, view_points, points_in_box
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box

from pathlib import Path

import copy
from pyquaternion import Quaternion

import cv2

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0

def map_pointcloud_to_image(pointsensor_token: str, camera_token: str):
    """Given a point sensor (lidar/radar) token and camera sample_data token, load point-cloud and map it to
    the image plane.

    Args:
        pointsensor_token: Lidar/radar sample_data token.
        camera_token: Camera sample_data token.

    Returns: (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).

    """

    cam = level5data.get("sample_data", camera_token)
    pointsensor = level5data.get("sample_data", pointsensor_token)
    pcl_path = level5data.data_path / pointsensor["filename"]
    pc = LidarPointCloud.from_file(pcl_path)
    image = Image.open(str(level5data.data_path / cam["filename"]))

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = level5data.get("calibrated_sensor", pointsensor["calibrated_sensor_token"])
    pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
    pc.translate(np.array(cs_record["translation"]))

    # Second step: transform to the global frame.
    poserecord = level5data.get("ego_pose", pointsensor["ego_pose_token"])
    pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
    pc.translate(np.array(poserecord["translation"]))

    # Third step: transform into the ego vehicle frame for the timestamp of the image.
    poserecord = level5data.get("ego_pose", cam["ego_pose_token"])
    pc.translate(-np.array(poserecord["translation"]))
    pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

    # Fourth step: transform into the camera.
    cs_record = level5data.get("calibrated_sensor", cam["calibrated_sensor_token"])
    pc.translate(-np.array(cs_record["translation"]))
    pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]

    # Retrieve the color from the depth.
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cs_record["camera_intrinsic"]), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < image.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < image.size[1] - 1)
    points = points[:, mask]
    points[2,:] = depths
    coloring = coloring[mask]

    return points

def extract_pc_in_box2d(pc, box2d):
    ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
    box2d_corners = np.zeros((4, 2))
    box2d_corners[0, :] = [box2d[0], box2d[1]]
    box2d_corners[1, :] = [box2d[2], box2d[1]]
    box2d_corners[2, :] = [box2d[2], box2d[3]]
    box2d_corners[3, :] = [box2d[0], box2d[3]]
    box2d_roi_inds = in_hull(pc[:, 0:2], box2d_corners)
    return pc[box2d_roi_inds, :], box2d_roi_inds

def extract_frustum_data(dataset, split = False , output_filename,
                         perturb_box2d=False, augmentX=1, type_whitelist=['Car'], remove_diff=False):
    level5data = LyftDataset(data_path=r"F:\LyftDataset\v1.01-train",
                             json_path=r"F:\LyftDataset\v1.01-train\v1.01-train",
                             verbose=True)

    id_list = []  # int number
    box2d_list = []  # [xmin,ymin,xmax,ymax]
    box3d_list = []  # (8,3) array in rect camera coord
    input_list = []  # channel number = 4, xyz,intensity in rect camera coord
    label_list = []  # 1 for roi object, 0 for clutter
    type_list = []  # string e.g. Car
    heading_list = []  # ry (along y-axis in rect camera coord) radius of
    # (cont.) clockwise angle from positive x axis in velo coord.
    box3d_size_list = []  # array of l,w,h
    frustum_angle_list = []  # angle of 2d box center from pos x-axis

    gt_box2d_list = []

    calib_list = []

    pos_cnt = 0
    all_cnt = 0
    idx = 0
    camera_channels = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']

    for scene in level5data.scene:
        cur_sample_token = scene["first_sample_token"]
        while cur_sample_token:
            print("[{0}] Current sample token: {1}".format(idx, cur_sample_token))
            my_sample = level5data.get('sample', cur_sample_token)

            if idx == 2758:
                cur_sample_token = my_sample['next']
                idx += 1
                continue
            lidar_data = level5data.get('sample_data', my_sample['data']['LIDAR_TOP'])
            pc = LidarPointCloud.from_file(Path(os.path.join('./', lidar_data['filename']))).points[:3, :]
            cs_lidar_record = level5data.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
            points = map_pointcloud_to_image(lidar_data, camera_token)


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
        save_dir = './pickles/'
    else:
        save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.car_only:
        type_whitelist = ['Car']
        output_prefix = 'frustum_car_only_'

    elif args.people_only:
        type_whitelist = ['Pedestrian', 'Cyclist']
        output_prefix = 'frustum_ped_cyc_'

    else:
        type_whitelist = ['Car', 'Pedestrian', 'Cyclist']
        output_prefix = 'frustum_car_ped_cyc_'

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

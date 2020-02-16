from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from PIL import Image
import pickle
import argparse
import json
from pathlib import Path

# Dataset helpers and loading utils -----------------------------------------------------------------


from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.geometry_utils import BoxVisibility, box_in_image, view_points, points_in_box, transform_matrix
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion

import copy
from pyquaternion import Quaternion

import cv2

level5data = LyftDataset(data_path=r'F:\\LyftDataset\\v1.01-train',
                         json_path=r'F:\\LyftDataset\\v1.01-train\\v1.01-train', verbose=True)


def load_table(filepath):
    with open(str(filepath)) as f:
        table = json.load(f)
    return table


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

    return pc.points, cam["filename"]


def read_2d_labels(img_name):
    path = r'F:\v1.01-train\Bbox2D'
    txt_name = img_name[7:-5] + '.txt'
    with open(os.path.join(path, txt_name), "r", encoding="utf-8") as f:
		content = eval(f.read())
    return content


def quaternion_yaw(q: Quaternion, in_image_frame: bool=True) -> float:

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
    box2d_roi_inds = in_hull(pc[:, 0:2], box2d_corners)
    return pc[box2d_roi_inds, :], box2d_roi_inds


def extract_frustum_data_rgb_detection(idx_filename, split, output_filename, res_label_dir,
                                       type_whitelist=['Car'],
                                       img_height_threshold=5,
                                       lidar_point_threshold=1):
    """ Extract point clouds in frustums extruded from 2D detection boxes.
        Update: Lidar points and 3d boxes are in *rect camera* coord system
            (as that in 3d box label files)

    Input:
        det_filename: string, each line is
            img_path typeid confidence xmin ymin xmax ymax
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        type_whitelist: a list of strings, object types we are interested in.
        img_height_threshold: int, neglect image with height lower than that.
        lidar_point_threshold: int, neglect frustum with too few points.
    Output:
        None (will write a .pickle file to the disk)
    """

    level5data = LyftDataset(data_path=r"F:\LyftDataset\v1.01-train",
                             json_path=r"F:\LyftDataset\v1.01-train\v1.01-train",
                             verbose=True)

    id_list = []
    type_list = []
    box2d_list = []
    prob_list = []
    input_list = []  # channel number = 4, xyz,intensity in rect camera coord
    frustum_angle_list = []  # angle of 2d box center from pos x-axis
    box3d_pred_list = []
	token_list = []
    calib_list = []
    enlarge_box3d_list = []
    enlarge_box3d_size_list = []
    enlarge_box3d_angle_list = []
    cam_channels = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    idx = 0

    for scene in level5data.scene:
        cur_sample_token = scene["first_sample_token"]
        while cur_sample_token:
            print("[{0}] Current sample token: {1}".format(idx, cur_sample_token))
            for channel in cam_channels:
                sample_record = level5data.get("sample", cur_sample_token)
                # Here we just grab the front camera and the point sensor.
                pointsensor_token = sample_record["data"]["LIDAR_TOP"]
                camera_token = sample_record["data"][channel]
                pc, image_name = map_pointcloud_to_image(pointsensor_token, camera_token)
                sample_annotation_tokens = sample_record['anns']
                Box_2D = read_2d_labels(image_name)
                for object,sample_annotation_token in zip(Box_2D, sample_annotation_tokens):
                    class_name = object[1]
                    min_x = float(object[0][0])
                    min_y = float(object[0][1])
                    max_x = float(object[0][2])
                    max_y = float(object[0][3])
					token = object[2]
                    box_2d = [min_x, min_y, max_x, max_y]
					uvdepth = np.zeros((1, 3))
					box2d_center = np.array([(min_x+max_x)/2.0,(min_y+max_y)/2.0])
					uvdepth[0, 0:2] = box2d_center
					uvdepth[0, 2] = 20
					level5data.get_sample_data(sample_cam_token, box_vis_level=BoxVisibility.ANY)

                    pc, box_2d_roi = extract_pc_in_box2d(pc, box_2d)
                    id_list.append(camera_token)
                    type_list.append(class_name)
                    box2d_list.append(box_2d)
                    prob_list.append(100)
					token_list.append(token)
                    sample_annotation = level5data.get('sample_annotation', sample_annotation_token)

	with open(output_filename, 'wb') as fp:
        pickle.dump(token_list, fp, -1)
        pickle.dump(box2d_list, fp, -1)
        pickle.dump(box3d_list, fp, -1)
        pickle.dump(input_list, fp, -1)
        pickle.dump(label_list, fp, -1)
        pickle.dump(type_list, fp, -1)
        pickle.dump(heading_list, fp, -1)
        pickle.dump(box3d_size_list, fp, -1)
        pickle.dump(frustum_angle_list, fp, -1)
        pickle.dump(gt_box2d_list, fp, -1)
        pickle.dump(calib_list, fp, -1)


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

# Work adapted to work for Lyft dataset from the original implementation of Frustum PoinNet by Qi et al.
# link to original implementation github repo: https://github.com/charlesq34/frustum-pointnets

from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from PIL import Image
import argparse
import pickle
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix, points_in_box, BoxVisibility
from lyft_utils import level5data
from lyft_utils import read_pointcloud, transform_pc_to_camera_coord, map_pointcloud_to_image, mask_points, \
    get_box_corners, get_2d_corners_from_projected_box_coordinates, random_shift_box2d, get_frustum_angle, \
    get_box_yaw_angle_in_camera_coords, read_det_file, extract_pc_in_box3d, extract_pc_in_box2d

from pkl_to_tfrec import tfrecGen_test, tfrec_Gen_Train_Val


num_scenes = len(level5data.scene)
clip_distance = 2.0
max_clip_distance = 60


def extract_frustum_data(scene_list, output_filename, viz=False, perturb_box2d=False, augmentX=5,
                         type_whitelist=['Car'], cam_channels=["CAM_FRONT"], txtfile = './train.txt'):
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

    vru_list = ['bicycle', 'pedestrian'] # To perform extra augmentation for countering data imbalance

    for scene in scene_list:
        print(scene)
        start_sample_token = level5data.scene[scene]['first_sample_token']
        sample_token = start_sample_token
        while sample_token != "":
            print(sample_token)
            sample_record = level5data.get("sample", sample_token)
            for channel in cam_channels:
                camera_token = sample_record["data"][channel]
                camera_data = level5data.get('sample_data', camera_token)

                f = open(txtfile, 'w')
                f.write(camera_token + '\n')
                f.close()

                point_cloud, lidar_token = read_pointcloud(sample_record, use_multisweep=False)

                image_path, box_list, cam_intrinsic = level5data.get_sample_data(camera_token,
                                                                                 box_vis_level=BoxVisibility.ALL,
                                                                                 selected_anntokens=None)
                img = Image.open(image_path)

                point_cloud_in_camera_coord_3d, point_cloud_in_camera_coord_2d = \
                    transform_pc_to_camera_coord(camera_data, level5data.get('sample_data', lidar_token),
                                                 point_cloud, level5data)
                for box in box_list:
                    if box.name not in type_whitelist:
                        continue
                    if box.name in vru_list:
                        augmentX = 6
                    for i in range(augmentX):
                        mask = mask_points(point_cloud_in_camera_coord_2d, 0, img.size[0], ymin=0, ymax=img.size[1])

                        distance_mask = (point_cloud.points[2, :] > clip_distance) & (
                                point_cloud.points[2, :] < max_clip_distance)

                        mask = np.logical_and(mask, distance_mask)

                        projected_corners_8pts = get_box_corners(box, cam_intrinsic,
                                                                 frustum_pointnet_convention=True)

                        xmin, xmax, ymin, ymax = get_2d_corners_from_projected_box_coordinates(projected_corners_8pts)
                        if perturb_box2d:
                            xmin, ymin, xmax, ymax = random_shift_box2d(xmin, xmax, ymin, ymax)

                        box_mask = mask_points(point_cloud_in_camera_coord_2d, xmin, xmax, ymin, ymax)
                        mask = np.logical_and(mask, box_mask)

                        point_clouds_in_box = point_cloud.points[:, mask]

                        _, seg_label = extract_pc_in_box3d(point_clouds_in_box[0:3, :], box.corners())

                        heading_angle = get_box_yaw_angle_in_camera_coords(box)
                        frustum_angle = get_frustum_angle(level5data, camera_token, xmax, xmin, ymax, ymin)
                        point_clouds_in_box = np.transpose(point_clouds_in_box)
                        box_2d_pts = np.array([xmin, ymin, xmax, ymax])
                        box_3d_pts = np.transpose(box.corners())
                        w, l, h = box.wlh
                        box3d_size = np.array([l, w, h])

                        if point_clouds_in_box.shape[0] < 100 and np.sum(seg_label) == 0:
                            continue

                        id_list.append(sample_record['token'])
                        box2d_list.append(box_2d_pts)
                        box3d_list.append(box_3d_pts)  # (8,3) array in rect camera coord
                        input_list.append(point_clouds_in_box)  # channel number = 4, xyz,intensity in rect camera coord
                        label_list.append(seg_label)  # 1 for roi object, 0 for clutter
                        type_list.append(box.name)  # string e.g. Car
                        heading_list.append(heading_angle)  # ry (along y-axis in rect camera coord) radius of
                        # (cont.) clockwise angle from positive x axis in velo coord.
                        box3d_size_list.append(box3d_size)  # array of l,w,h
                        frustum_angle_list.append(frustum_angle)  # angle of 2d box center from pos x-axis

            next_sample_token = sample_record['next']
            sample_token = next_sample_token

    with open(output_filename, 'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list, fp)
        pickle.dump(box3d_list, fp)
        pickle.dump(input_list, fp)
        pickle.dump(label_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(heading_list, fp)
        pickle.dump(box3d_size_list, fp)
        pickle.dump(frustum_angle_list, fp)


def extract_frustum_data_rgb_detection(scene_list, det_filename, output_filename, cam_channels=["CAM_FRONT"],
                                       type_whitelist=['Car'], img_height_threshold=25, lidar_point_threshold=5,
                                       txtfile = './test.txt'):
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
    cache_id = -1
    cache = None

    id_list = []
    type_list = []
    box2d_list = []
    prob_list = []
    input_list = []  # channel number = 4, xyz,intensity in rect camera coord
    frustum_angle_list = []  # angle of 2d box center from pos x-axis
    detections_2d = read_det_file(det_filename)
    for scene in scene_list:
        start_sample_token = level5data.scene[scene]['first_sample_token']
        sample_token = start_sample_token
        while sample_token != "":
            print(sample_token)
            sample_record = level5data.get("sample", sample_token)
            for channel in cam_channels:

                pointsensor_token = sample_record["data"]["LIDAR_TOP"]
                camera_token = sample_record["data"][channel]

                if camera_token in detections_2d:
                    det_boxes = detections_2d[camera_token]
                else:
                    continue

                f = open(txtfile, 'w')
                f.write(camera_token+'\n')  # python will convert \n to os.linesep
                f.close()

                camera_data = level5data.get('sample_data', camera_token)

                point_cloud, lidar_token = read_pointcloud(sample_record, lyftd=level5data, use_multisweep=False)

                image_path, box_list, cam_intrinsic = level5data.get_sample_data(camera_token,
                                                                                 box_vis_level=BoxVisibility.ALL,
                                                                                 selected_anntokens=None)

                points, mask, image = map_pointcloud_to_image(pointsensor_token, camera_token)

                point_cloud_in_camera_coord_3d, point_cloud_in_camera_coord_2d = \
                    transform_pc_to_camera_coord(camera_data, level5data.get('sample_data', lidar_token),
                                                 point_cloud, level5data)

                for box in det_boxes:
                    obj_type = box[0]
                    prob = float(box[1])
                    box_2d = box[2]
                    if obj_type not in type_whitelist:
                        continue
                    xmin, ymin, xmax, ymax = box_2d[0], box_2d[1], box_2d[2], box_2d[3]
                    box_mask = mask_points(point_cloud_in_camera_coord_2d, xmin, xmax, ymin, ymax)
                    mask = np.logical_and(mask, box_mask)

                    point_clouds_in_box = point_cloud.points[:, mask]

                    frustum_angle = get_frustum_angle(level5data, camera_token, xmax, xmin, ymax, ymin)

                    if ymax - ymin < img_height_threshold or len(point_clouds_in_box) < lidar_point_threshold:
                        continue

                    id_list.append(camera_token)
                    type_list.append(obj_type)
                    box2d_list.append(box_2d)
                    prob_list.append(prob)
                    input_list.append(point_clouds_in_box)
                    frustum_angle_list.append(frustum_angle)

            next_sample_token = sample_record['next']
            sample_token = next_sample_token

    with open(output_filename, 'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list, fp)
        pickle.dump(input_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(frustum_angle_list, fp)
        pickle.dump(prob_list, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run demo.')
    parser.add_argument('--gen_train', action='store_true',
                        help='Generate train split frustum data with perturbed GT 2D boxes')
    parser.add_argument('--gen_val', action='store_true', help='Generate val split frustum data with GT 2D boxes')
    parser.add_argument('--gen_test', action='store_true', help='Generate val split frustum data with GT 2D boxes')
    parser.add_argument('--gen_rgb_detection_val', action='store_true',
                        help='Generate test split frustum data with GT 2D boxes')
    parser.add_argument('--num_scenes', type=int, choices=range(0, 180),
                        help='Only generate cars; otherwise cars, peds and cycs')
    parser.add_argument('--num_cam_channels', type=int, choices=0,
                        help='0 for only CAM_FRONT and other value for rest of cam_channels')

    args = parser.parse_args()

    num_scenes = args.num_scenes
    num_cam_channels = args.num_cam_channels
    num_train_scenes = int(0.5 * num_scenes)
    num_val_scenes = int(0.25 * num_scenes)
    num_test_scenes = int(0.25 * num_scenes)

    scenes = level5data.scene
    data_dir = '/scratch/jbandl2s/Lyft_dataset'
    det_filename = './rgb_detections.txt'

    type_whitelist = ['car', 'pedestrian', 'bicycle']
    # As we chose to perform 3D object detection of only these 3 classes

    if num_cam_channels == 0:
        cam_channels = ["CAM_FRONT"]
        prefix = 'Lyft_front'
    else:
        cam_channels = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
        prefix = 'Lyft'

    train_scenes = np.arange(0, num_train_scenes)
    val_scenes = np.arange(num_train_scenes, num_train_scenes + num_val_scenes)
    test_scenes = np.arange(num_train_scenes + num_val_scenes, num_scenes)

    print(train_scenes)
    print(val_scenes)
    print(test_scenes)

    if args.gen_train:
        output_prefix = prefix+'_train.pickle'
        extract_frustum_data(scene_list=train_scenes, type_whitelist=type_whitelist, viz=False, perturb_box2d=True,
                             augmentX=3, output_filename=os.path.join(data_dir, output_prefix),
                             cam_channels=cam_channels, txtfile = './train.txt')

    if args.gen_val:
        output_prefix = prefix+'_val.pickle'
        extract_frustum_data(scene_list=val_scenes, type_whitelist=type_whitelist, viz=False, perturb_box2d=True,
                             augmentX=1, output_filename=os.path.join(data_dir, output_prefix),
                             cam_channels=cam_channels, txtfile = './val.txt')

    if args.gen_test:
        output_prefix = prefix+'_test.pickle'
        extract_frustum_data(scene_list=test_scenes, type_whitelist=type_whitelist, viz=False, perturb_box2d=True,
                             augmentX=1, output_filename=os.path.join(data_dir, output_prefix),
                             cam_channels=cam_channels, txtfile = './test.txt')

    if args.gen_rgb_detection_val:
        output_prefix = 'half_front_val_rgb_detection.pickle'
        extract_frustum_data_rgb_detection(scene_list=val_scenes, output_filename=os.path.join(data_dir, output_prefix),
                                           type_whitelist=type_whitelist, cam_channels=cam_channels,
                                           det_filename=det_filename)

    # For ease of working with Keras based Frustum PointNet, we converted the pickles generated above to tfrec files
    output_prefix = prefix + '_train.pickle'
    test_gen = tfrec_Gen_Train_Val(os.path.join(data_dir, output_prefix))
    test_gen.write_tfrec('/scratch/jbandl2s/Lyftdataset/lyft_train.tfrec')

    output_prefix = prefix + '_val.pickle'
    test_gen = tfrec_Gen_Train_Val(os.path.join(data_dir, output_prefix))
    test_gen.write_tfrec('/scratch/jbandl2s/Lyftdataset/lyft_val.tfrec')

    output_prefix = prefix + '_test.pickle'
    test_gen = tfrecGen_test(os.path.join(data_dir, output_prefix))
    test_gen.write_tfrec('/scratch/jbandl2s/Lyftdataset/lyft_test.tfrec')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
import cv2
import os
from datetime import datetime
from config import cfg
import pickle

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix


def Extract_Pointcloud(train_df):
    sample_token = train_df.first_sample_token.values[0]
    visualize_lidar_of_sample(sample_token)
    sample = Ll5dataset.get("sample", sample_token)

    sample_lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = Ll5dataset.get("sample_data", sample_lidar_token)
    lidar_filepath = Ll5dataset.get_sample_data_path(sample_lidar_token)
    lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)

    return lidar_pointcloud, lidar_data


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds


def extract_pc_in_box2d(pc, box2d):
    ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
    box2d_corners = np.zeros((4,2))
    box2d_corners[0,:] = [box2d[0],box2d[1]]
    box2d_corners[1,:] = [box2d[2],box2d[1]]
    box2d_corners[2,:] = [box2d[2],box2d[3]]
    box2d_corners[3,:] = [box2d[0],box2d[3]]
    box2d_roi_inds = in_hull(pc[:,0:2], box2d_corners)
    return pc[box2d_roi_inds,:], box2d_roi_inds

def Extract_Images(train_df, channel_to_get):
    sample_token = train_df.first_sample_token.values[0]
    sample = Ll5dataset.get("sample", sample_token)
    sample_camera_token = sample['data'][channel_to_get]
    return Ll5dataset.get('sample_data', sample_camera_token)


def Extract_Ego_Pose(sensorInfo):
    ego_pose = Ll5dataset.get("ego_pose", sensorInfo["ego_pose_token"])
    calibrated_sensor = Ll5dataset.get("calibrated_sensor", sensorInfo["calibrated_sensor_token"])
    return ego_pose, calibrated_sensor


# def get_data_from_sample(train_df,channel_to_get):
#     sample_token = train_df.first_sample_token.values[0]
#     sample = Ll5dataset.get("sample", sample_token)
#     return Ll5dataset.get('sample_data', sample['data'][channel_to_get])


def Extract_Trans_Matrix(ego_pose, calibration_mat):
    global_from_car = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=False)
    car_from_sensor = transform_matrix(calibration_mat['translation'], Quaternion(calibration_mat['rotation']),
                                       inverse=False)
    return global_from_car, car_from_sensor


def show_img_from_data(data):
    plt.imshow(
        cv2.cvtColor(
            cv2.imread(data['filename']),
            cv2.COLOR_BGR2RGB
        )
    )


def extract_frustum_data(token, split = False, output_filename, viz=False,
                         perturb_box2d=False, augmentX=1, type_whitelist=['Car','Cyclist','Pedestrian']):
    ''' Extract point clouds and corresponding annotations in frustums
        defined generated from 2D bounding boxes
        Lidar points and 3d boxes are in *rect camera* coord system
        (as that in 3d box label files)

    Input:
        idx_filename: string, each line of the file is a sample ID
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        viz: bool, whether to visualize extracted data
        perturb_box2d: bool, whether to perturb the box2d
            (used for data augmentation in train set)
        augmentX: scalar, how many augmentations to have for each 2D box.
        type_whitelist: a list of strings, object types we are interested in.
    Output:
        None (will write a .pickle file to the disk)
    '''
    # dataset = kitti_object(os.path.join(ROOT_DIR,'dataset/KITTI/object'), split)
    dataset = kitti_object(KITTI_DATA_PATH, split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

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

    pos_cnt = 0
    all_cnt = 0
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx)  # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        pc_velo = dataset.get_lidar(data_idx)
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
        pc_rect[:, 3] = pc_velo[:, 3]
        img = dataset.get_image(data_idx)
        img_height, img_width, img_channel = img.shape
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:, 0:3],
                                                                 calib, 0, 0, img_width, img_height, True)

        for obj_idx in range(len(objects)):
            if objects[obj_idx].type not in type_whitelist: continue

            # 2D BOX: Get pts rect backprojected
            box2d = objects[obj_idx].box2d
            for _ in range(augmentX):
                # Augment data by box2d perturbation
                if perturb_box2d:
                    xmin, ymin, xmax, ymax = random_shift_box2d(box2d)
                    print(box2d)
                    print(xmin, ymin, xmax, ymax)
                else:
                    xmin, ymin, xmax, ymax = box2d
                box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
                               (pc_image_coord[:, 0] >= xmin) & \
                               (pc_image_coord[:, 1] < ymax) & \
                               (pc_image_coord[:, 1] >= ymin)
                box_fov_inds = box_fov_inds & img_fov_inds
                pc_in_box_fov = pc_rect[box_fov_inds, :]
                # Get frustum angle (according to center pixel in 2D BOX)
                box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
                uvdepth = np.zeros((1, 3))
                uvdepth[0, 0:2] = box2d_center
                uvdepth[0, 2] = 20  # some random depth
                box2d_center_rect = calib.project_image_to_rect(uvdepth)
                frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                                box2d_center_rect[0, 0])
                # 3D BOX: Get pts velo in 3d box
                obj = objects[obj_idx]
                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
                _, inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
                label = np.zeros((pc_in_box_fov.shape[0]))
                label[inds] = 1
                # Get 3D BOX heading
                heading_angle = obj.ry
                # Get 3D BOX size
                box3d_size = np.array([obj.l, obj.w, obj.h])

                # Reject too far away object or object without points
                if ymax - ymin < 25 or np.sum(label) == 0:
                    continue

                id_list.append(data_idx)
                box2d_list.append(np.array([xmin, ymin, xmax, ymax]))
                box3d_list.append(box3d_pts_3d)
                input_list.append(pc_in_box_fov)
                label_list.append(label)
                type_list.append(objects[obj_idx].type)
                heading_list.append(heading_angle)
                box3d_size_list.append(box3d_size)
                frustum_angle_list.append(frustum_angle)

                # collect statistics
                pos_cnt += np.sum(label)
                all_cnt += pc_in_box_fov.shape[0]

    print('Average pos ratio: %f' % (pos_cnt / float(all_cnt)))
    print('Average npoints: %f' % (float(all_cnt) / len(id_list)))

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


def visualize_lidar_of_sample(sample_token, axes_limit=80):
    sample = Ll5dataset.get("sample", sample_token)
    sample_lidar_token = sample["data"]["LIDAR_TOP"]
    Ll5dataset.render_sample_data(sample_lidar_token, axes_limit=axes_limit)


if __name__ == '__main__':

    data_path = cfg.DATA.DATA_PATH
    json_path = cfg.DATA.TRAIN_JSON_PATH
    dataset_path: str = 'F:/LyftDataset/v1.01-train/'

    Ll5dataset = LyftDataset(data_path=dataset_path, json_path=dataset_path + 'v1.01-train/', verbose=True)
    os.makedirs(dataset_path + 'Artifacts/', exist_ok=True)

    records = [(Ll5dataset.get('sample', record['first_sample_token'])['timestamp'], record) for record in
               Ll5dataset.scene]
    entries = []

    for start_time, record in sorted(records):
        start_time = Ll5dataset.get('sample', record['first_sample_token'])['timestamp'] / 1000000
        token = record['token']
        name = record['name']
        date = datetime.utcfromtimestamp(start_time)
        host = "-".join(record['name'].split("-")[:2])
        first_sample_token = record["first_sample_token"]
        entries.append((host, name, date, token, first_sample_token))

    df = pd.DataFrame(entries, columns=["host", "scene_name", "date", "scene_token", "first_sample_token"])
    validation_hosts = ["host-a007", "host-a008", "host-a009"]
    validation_df = df[df["host"].isin(validation_hosts)]
    vi = validation_df.index
    train_df = df[~df.index.isin(vi)]
    classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal",
               "emergency_vehicle"]
    LidarPointCloud, LidarSensorInformation = Extract_Pointcloud(train_df)
    ego_pose, calibration_mat = Extract_Ego_Pose(LidarSensorInformation)
    global_from_car, car_from_sensor = Extract_Trans_Matrix(ego_pose, calibration_mat)
    camera1_channel = 'CAM_BACK'
    camera2_channel = 'CAM_BACK_LEFT'
    camera3_channel = 'CAM_FRONT'
    camera4_channel = 'CAM_FRONT_ZOOMED'
    camera5_channel = 'CAM_FRONT_LEFT'
    camera6_channel = 'CAM_FRONT_RIGHT'
    camera7_channel = 'CAM_BACK_RIGHT'
    Back_Image = Extract_Images(train_df, camera1_channel)
    Back_Left_Image = Extract_Images(train_df, camera2_channel)
    Back_Right_Image = Extract_Images(train_df, camera7_channel)
    Front_Image = Extract_Images(train_df, camera3_channel)
    Front_Left_Image = Extract_Images(train_df, camera5_channel)
    Front_Right_Image = Extract_Images(train_df, camera6_channel)
    # show_img_from_data(Front_Image)

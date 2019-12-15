import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
import cv2
import os
from datetime import datetime
from config import cfg

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

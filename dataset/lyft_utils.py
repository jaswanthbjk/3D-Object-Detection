from __future__ import print_function

import os
import sys

import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion, RadarPointCloud
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.geometry_utils import view_points

level5data = LyftDataset(data_path='/scratch/jbandl2s/Lyft_dataset/Lyft_train/',
                         json_path='/scratch/jbandl2s/Lyft_dataset/Lyft_train/data/',
                         verbose=True)


def read_pointcloud(sample_record, lyftd=level5data, use_multisweep=False):
    lidar_data_token = sample_record['data']['LIDAR_TOP']
    lidar_data_record = lyftd.get("sample_data", lidar_data_token)

    pcl_path = lyftd.get_sample_data_path(lidar_data_token)

    if use_multisweep:
        pc, _ = LidarPointCloud.from_file_multisweep(level5data, sample_record, chan='LIDAR_TOP',
                                                     ref_chan='LIDAR_TOP', num_sweeps=26)
    else:
        pc = LidarPointCloud.from_file(pcl_path)

    return pc, lidar_data_token


def transform_pc_to_camera_coord(cam: dict, pointsensor: dict, point_cloud_3d: LidarPointCloud, lyftd: level5data):
    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = lyftd.get("calibrated_sensor", pointsensor["calibrated_sensor_token"])
    point_cloud_3d.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
    point_cloud_3d.translate(np.array(cs_record["translation"]))
    # Second step: transform to the global frame.
    poserecord = lyftd.get("ego_pose", pointsensor["ego_pose_token"])
    point_cloud_3d.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
    point_cloud_3d.translate(np.array(poserecord["translation"]))
    # Third step: transform into the ego vehicle frame for the timestamp of the image.
    poserecord = lyftd.get("ego_pose", cam["ego_pose_token"])
    point_cloud_3d.translate(-np.array(poserecord["translation"]))
    point_cloud_3d.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)
    # Fourth step: transform into the camera.
    cs_record = lyftd.get("calibrated_sensor", cam["calibrated_sensor_token"])
    point_cloud_3d.translate(-np.array(cs_record["translation"]))
    point_cloud_3d.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    point_cloud_2d = view_points(point_cloud_3d.points[:3, :],
                                 np.array(cs_record["camera_intrinsic"]), normalize=True)

    return point_cloud_3d, point_cloud_2d


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
    if pointsensor["sensor_modality"] == "lidar":
        pc = LidarPointCloud.from_file(pcl_path)
    else:
        pc = RadarPointCloud.from_file(pcl_path)
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
    coloring = coloring[mask]

    return pc.points, mask, image


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    """ pc: (N,3), box3d: (8,3) """
    assert box3d.shape == (3, 8)
    assert pc.shape[0] == 3
    pc = np.transpose(pc)
    box3d = np.transpose(box3d)

    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


def extract_pc_in_box2d(pc, box2d):
    ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
    box2d_corners = np.zeros((4, 2))
    box2d_corners[0, :] = [box2d[0], box2d[1]]
    box2d_corners[1, :] = [box2d[2], box2d[1]]
    box2d_corners[2, :] = [box2d[2], box2d[3]]
    box2d_corners[3, :] = [box2d[0], box2d[3]]
    box2d_roi_inds = in_hull(pc[:, 0:2], box2d_corners)
    return pc[box2d_roi_inds, :], box2d_roi_inds


def mask_points(points: np.ndarray, xmin,
                xmax, ymin, ymax, depth_min=0, buffer_pixel=1) -> np.ndarray:
    """
    Mask out points outside xmax,xmin,ymin,ymax


    :param points:
    :param xmin:
    :param xmax:
    :param ymin:
    :param ymax:
    :param depth_min:
    :param buffer_pixel:
    :return: index array
    """
    depths = points[2, :]

    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > depth_min)
    mask = np.logical_and(mask, points[0, :] > xmin + buffer_pixel)
    mask = np.logical_and(mask, points[0, :] < xmax - buffer_pixel)
    mask = np.logical_and(mask, points[1, :] > ymin + buffer_pixel)
    mask = np.logical_and(mask, points[1, :] < ymax - buffer_pixel)

    return mask


def get_box_corners(transformed_box: Box, cam_intrinsic_mtx: np.array, frustum_pointnet_convention=True):
    box_corners_on_cam_coord = transformed_box.corners()

    # Rearrange to conform Frustum-pointnet's convention

    if frustum_pointnet_convention:
        rearranged_idx = [0, 3, 7, 4, 1, 2, 6, 5]
        box_corners_on_cam_coord = box_corners_on_cam_coord[:, rearranged_idx]

        assert np.allclose((box_corners_on_cam_coord[:, 0] + box_corners_on_cam_coord[:, 6]) / 2,
                           np.array(transformed_box.center))

    # For perspective transformation, the normalization should set to be True
    box_corners_on_image = view_points(box_corners_on_cam_coord, view=cam_intrinsic_mtx, normalize=True)

    return box_corners_on_image


def get_2d_corners_from_projected_box_coordinates(projected_corners: np.ndarray):
    assert projected_corners.shape[0] == 3

    xmin = projected_corners[0, :].min()
    xmax = projected_corners[0, :].max()
    ymin = projected_corners[1, :].min()
    ymax = projected_corners[1, :].max()

    return xmin, xmax, ymin, ymax


def random_shift_box2d(xmin, xmax, ymin, ymax, shift_ratio=0.1):
    """ Randomly shift box center, randomly scale width and height
    """
    r = shift_ratio
    h = ymax-ymin
    w = xmax-xmin
    cx = (xmin+xmax)/2.0
    cy = (ymin+ymax)/2.0
    cx2 = cx + w*r*(np.random.random()*2-1)
    cy2 = cy + h*r*(np.random.random()*2-1)
    h2 = h*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    w2 = w*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    return np.array([cx2-w2/2.0, cy2-h2/2.0, cx2+w2/2.0, cy2+h2/2.0])


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


def get_box_yaw_angle_in_camera_coords(box: Box):
    """
    Calculate the heading angle, using the convention in KITTI labels.

    :param box: bouding box
    :return:
    """

    box_corners = box.corners()
    v = box_corners[:, 0] - box_corners[:, 4]
    heading_angle = np.arctan2(-v[2], v[0])
    return heading_angle


def get_frustum_angle(lyftd: level5data, cam_token, xmax, xmin, ymax, ymin):
    random_depth = 20
    image_center = np.array([[(xmax + xmin) / 2, (ymax + ymin) / 2, random_depth]]).T
    image_center_in_cam_coord = transform_image_to_cam_coordinate(image_center, cam_token, lyftd)
    assert image_center_in_cam_coord.shape[1] == 1
    frustum_angle = -np.arctan2(image_center_in_cam_coord[2, 0], image_center_in_cam_coord[0, 0])
    return frustum_angle


def read_det_file(det_filename):
    """ Parse lines in 2D detection output files """
    det_id2str = {0:'animal',1:'bicycle',2:'bus',3:'car',4:'emergency_vehicle',5:'motorcycle',6:'other_vehicle',
                  7:'pedestrian',8:'truck'}
    prev_id = ""
    detections = {}
    for line in open(det_filename, 'r'):
        t = line.rstrip().split(" ")
        if t[0] == prev_id:
            detections[t[0]].append([t[1], t[2], np.array([float(t[i]) for i in range(3, 7)])])
        else:
            detections[t[0]] = [[t[1], t[2], np.array([float(t[i]) for i in range(3, 7)])]]
            prev_id = t[0]

    return detections

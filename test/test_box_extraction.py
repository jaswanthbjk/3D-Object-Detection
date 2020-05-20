import unittest
import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np

from dataset.prepare_lyft_data import extract_single_box, \
    parse_train_csv, level5data, extract_boxed_clouds, \
    get_sample_images, get_train_data_sample_token_and_box, \
    get_pc_in_image_fov, transform_bounding_box_to_sensor_coord_and_get_corners, \
    get_2d_corners_from_projected_box_coordinates, get_sensor_to_world_transform_matrix, transform_box_from_world_to_sensor_coordinates
from dataset.prepare_lyft_data_v2 import transform_image_to_cam_coordinate
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
from lyft_dataset_sdk.utils.geometry_utils import view_points


class MyTestCase(unittest.TestCase):

    def test_extract_box(self):
        train_df = parse_train_csv()

        box, sample_data_token = extract_single_box(train_df, 0)

        sample_token = "92bff46db1dbfc9679edc8091770c4256ac3c027e9f0a9c31dfc4fff41f6f677"
        box_from_annotation_token = level5data.get_box(sample_token)

        # self.assertEqual(box,box_from_annotation_token)

    def test_extract_box_clouds(self):
        a, b = extract_boxed_clouds(100)

        print(a.shape)
        print(b.shape)

    def test_get_sample_images(self):
        # train_df = parse_train_csv()

        # sample_token=train_df.iloc[0,0]
        # print(sample_token)
        sample_token = "db8b47bd4ebdf3b3fb21598bb41bd8853d12f8d2ef25ce76edd4af4d04e49341"

        get_sample_images(sample_token)

    def test_get_sample_image_and_transform_box(self):
        train_df = parse_train_csv()
        sample_token, bounding_box = get_train_data_sample_token_and_box(0, train_df)

        first_train_sample = level5data.get('sample', sample_token)

        lidar_data_token = first_train_sample['data']['LIDAR_TOP']

        mask, lpc_array_in_cam_coord, filtered_pc_2d, _, image = get_pc_in_image_fov(lidar_data_token, 'CAM_FRONT',
                                                                                     bounding_box)

        plt.scatter(lpc_array_in_cam_coord[0, :], lpc_array_in_cam_coord[2, :])

        fig, ax = plt.subplots(1, 1)
        ax.imshow(image)
        print(filtered_pc_2d.shape)
        ax.plot(filtered_pc_2d[0, :], filtered_pc_2d[1, :], '.')
        plt.show()

    def test_transform_image_to_camera_coord(self):
        train_df = parse_train_csv()
        sample_token, bounding_box = get_train_data_sample_token_and_box(0, train_df)
        first_train_sample = level5data.get('sample', sample_token)

        cam_token = first_train_sample['data']['CAM_FRONT']
        sd_record = level5data.get("sample_data", cam_token)
        cs_record = level5data.get("calibrated_sensor", sd_record["calibrated_sensor_token"])

        cam_intrinsic_mtx = np.array(cs_record["camera_intrinsic"])

        box_corners = transform_bounding_box_to_sensor_coord_and_get_corners(bounding_box, cam_token)

        # check)image
        cam_image_file = level5data.get_sample_data_path(cam_token)
        cam_image_mtx = imread(cam_image_file)

        xmin, xmax, ymin, ymax = get_2d_corners_from_projected_box_coordinates(box_corners)

        random_depth = 20
        image_center = np.array([[(xmax + xmin) / 2, (ymax + ymin) / 2, random_depth]]).T

        image_center_in_cam_coord = transform_image_to_cam_coordinate(image_center, cam_token)

        self.assertTrue(np.isclose(random_depth, image_center_in_cam_coord[2:]))

        transformed_back_image_center = view_points(image_center_in_cam_coord, cam_intrinsic_mtx, normalize=True)

        self.assertTrue(np.allclose(image_center[0:2, :], transformed_back_image_center[0:2, :]))

    def test_get_bounding_box_corners(self):
        train_df = parse_train_csv()
        sample_token, bounding_box = get_train_data_sample_token_and_box(0, train_df)
        first_train_sample = level5data.get('sample', sample_token)

        cam_token = first_train_sample['data']['CAM_FRONT']

        box_corners = transform_bounding_box_to_sensor_coord_and_get_corners(bounding_box, cam_token)

        print(box_corners)

        # check image
        cam_image_file = level5data.get_sample_data_path(cam_token)
        cam_image_mtx = imread(cam_image_file)

        xmin, xmax, ymin, ymax = get_2d_corners_from_projected_box_coordinates(box_corners)

        fig, ax = plt.subplots(1, 1)
        ax.imshow(cam_image_mtx)
        ax.plot(box_corners[0, :], box_corners[1, :])
        ax.plot([xmin, xmin, xmax, xmax], [ymin, ymax, ymin, ymax])
        plt.show()

    def test_convert_back_world_coordinate(self):
        train_df = parse_train_csv()
        sample_token, bounding_box = get_train_data_sample_token_and_box(0, train_df)

        # sample_token="db8b47bd4ebdf3b3fb21598bb41bd8853d12f8d2ef25ce76edd4af4d04e49341"
        train_sample = level5data.get('sample', sample_token)

        cam_token = train_sample['data']['CAM_FRONT']

        bounding_box_in_sensor_coord = transform_box_from_world_to_sensor_coordinates(bounding_box, cam_token, )

        transform_mtx = get_sensor_to_world_transform_matrix(sample_token, 'CAM_FRONT')

        box_corner_2d, box_corner_3d = transform_bounding_box_to_sensor_coord_and_get_corners(bounding_box,
                                                                                              cam_token, frustum_pointnet_convention=True)

        box_center = (box_corner_3d[:, 0] + box_corner_3d[:, 6]) / 2
        print(box_center)
        print(bounding_box_in_sensor_coord.center)

        box_center_h = np.concatenate((box_center, np.ones(1)))

        box_center_world = np.dot(transform_mtx, box_center_h.T)

        print(box_center_world)
        print(bounding_box.center)


if __name__ == '__main__':
    unittest.main()

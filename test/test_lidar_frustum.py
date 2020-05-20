import matplotlib.pyplot as plt

from helpers.viz_util_for_lyft import draw_lidar_simple
from dataset.prepare_lyft_data import get_pc_in_image_fov
from dataset.prepare_lyft_data_v2 import load_train_data


def test_draw_lidar_in_fov():
    lyftd = load_train_data()

    demo_lidar_sample_data_token = "d6e7198659531b587b42cfffed4dfcf2d326b2c5c9ac86fba2121948ddae2274"
    pc_fov, image = get_pc_in_image_fov(demo_lidar_sample_data_token, 'CAM_FRONT', lyftd=lyftd)

    draw_lidar_simple(pc_fov)

    input()

    plt.imshow(image)
    plt.show()


test_draw_lidar_in_fov()

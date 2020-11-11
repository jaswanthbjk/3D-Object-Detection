#!/usr/bin/python3
import os

import matplotlib.pyplot as plt
from prepare_lyft_data_v2 import load_train_data
from dataset.lyft_utils import level5data as load_train_data
from viz_util_for_lyft import PredViewer


def test_one_sample_token(sample_token, count, ldf, folder):
    pv = PredViewer(pred_file="./results_output.csv", lyftd=ldf)
    test_token = sample_token
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(18, 24))
    pv.render_camera_image(ax[0], sample_token=test_token, prob_threshold=0.7)
    pv.render_lidar_points(ax[1], sample_token=test_token, prob_threshold=0.7)
    plt.savefig(folder + "/combiimage%02d.png" % count)



def gen_image_test(sample_token, count, ldf, folder):
    pv = PredViewer(pred_file="./results_output.csv", lyftd=ldf)
    test_token = sample_token
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 24))
    pv.render_camera_image(ax, sample_token=test_token, prob_threshold=0.7)

    plt.savefig(folder + "/image%02d.png" % count)


def test_one_scene(scene_number):
    ldf = load_train_data()
    pv = PredViewer(pred_file="./results_output.csv", lyftd=ldf)
    test_token = pv.pred_pd.index[0]
    img = []  # some array of images
    frames = []
    fig, ax = plt.subplots(nrows=2, ncols=1)
    counter = 0
    start_sample_token = ldf.scene[scene_number]['first_sample_token']
    sample_token = start_sample_token
    while sample_token != " ":
        fig, ax = plt.subplots(nrows=2, ncols=1)
        sample_record = ldf.get('sample', sample_token)
        pv.render_camera_image(ax[0], sample_token=sample_token, prob_threshold=0.7)
        pv.render_lidar_points(ax[1], sample_token=sample_token, prob_threshold=0.7)
        plt.savefig('./artifacts' + "/image%02d.png" % counter)
        next_sample_token = sample_record['next']
        sample_token = next_sample_token
        counter += 1


def test_3d_lidar_points():
    pv = PredViewer(pred_file="./results_output.csv", lyftd=load_train_data())
    test_token = pv.pred_pd.index[0]

    pv.render_3d_lidar_points(sample_token=test_token)


def test_3d_lidar_points_in_camera_coords():
    pv = PredViewer(pred_file="./results_output.csv", lyftd=load_train_data())
    test_token = pv.pred_pd.index[0]

    pv.render_3d_lidar_points_to_camera_coordinates(test_token, prob_threshold=0.4)


def plot_prediction_data():
    lyftd = load_train_data()
    pv = PredViewer(pred_file=r"./results_output.csv", lyftd=lyftd)

    test_token = pv.pred_pd.index[1]

    pv.render_3d_lidar_points_to_camera_coordinates(test_token, prob_threshold=0.4)


if __name__ == "__main__":
    lyftD = load_train_data()
    counter = 0
    while input("Enter c to continue or e to exit: ") != 'e':
        scene_number = int(input('Enter Scene number: '))
        if not os.path.exists('./scene%03d/' % scene_number):
            os.makedirs('./scene%03d/' % scene_number)
        folder_path = './scene%03d/' % scene_number
        start_sample_token = lyftD.scene[scene_number]['first_sample_token']
        sample_token = start_sample_token
        while sample_token != " ":
            sample_record = lyftD.get('sample', sample_token)
            test_one_sample_token(sample_token, counter, ldf=lyftD, folder=folder_path)
            gen_image_test(sample_token, counter, ldf=lyftD, folder=folder_path)
            next_sample_token = sample_record['next']
            sample_token = next_sample_token
            counter += 1
            print(counter)

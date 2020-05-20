"""
This file visualizes the boxes and lidar points on a image

"""

from helpers.viz_util_for_lyft import PredViewer
from dataset.prepare_lyft_data_v2 import load_train_data
import matplotlib.pyplot as plt

import mayavi.mlab as mlab

import os

FIG_PATH = "./"

token_id = 65

pv = PredViewer(pred_file="./log_pred_v2_01.csv", lyftd=load_train_data())
test_token = pv.pred_pd.index[token_id]

fig, ax = plt.subplots(nrows=2, ncols=1)
pv.render_camera_image(ax[0], sample_token=test_token, prob_threshold=0.4)

pv.render_lidar_points(ax[1], sample_token=test_token, prob_threshold=0.4)

plt.savefig(os.path.join(FIG_PATH, "demo_image_{}.png".format(token_id)))

# pv.render_lidar_points(ax[1], sample_token=test_token, prob_threshold=0.4)

mlab_fig = pv.render_3d_lidar_points_to_camera_coordinates(test_token, prob_threshold=0.4)

mlab.savefig(os.path.join(FIG_PATH, "demo_3d_lidar_{}.png".format(token_id)))

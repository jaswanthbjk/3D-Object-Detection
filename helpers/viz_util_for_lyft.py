''' Visualization code for point clouds and 3D bounding boxes with mayavi.

Modified by Charles R. Qi
Date: September 2017

Ref: https://github.com/hengck23/didi-udacity-2017/blob/master/baseline-04/kitti_data/draw.py
'''

import warnings
import numpy as np

try:
    import mayavi.mlab as mlab
except ImportError:
    warnings.warn("mayavi is not installed")
import pandas as pd
from dataset.prepare_lyft_data import parse_string_to_box, transform_box_from_world_to_sensor_coordinates, \
    get_sensor_to_world_transform_matrix_from_sample_data_token
from dataset.prepare_lyft_data_v2 import transform_pc_to_camera_coord
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
from lyft_dataset_sdk.utils.geometry_utils import box_in_image,BoxVisibility
from skimage.io import imread
import matplotlib.pyplot as plt


class PredViewer(object):

    def __init__(self, pred_file, lyftd: LyftDataset):
        self.pred_pd = pd.read_csv(pred_file, index_col="Id")
        self.lyftd = lyftd

    def get_boxes_from_token(self, sample_token):
        boxes_str = self.pred_pd.loc[sample_token, 'PredictionString']
        sample_token=sample_token
        boxes = parse_string_to_box(boxes_str,sample_token=sample_token)
        return boxes

    def get_sample_record_from_token(self, sample_token):
        pass

    def render_camera_image(self, ax, sample_token, cam_key='CAM_FRONT', prob_threshold=0.7):
        sample_record = self.lyftd.get('sample', sample_token)
        camera_token = sample_record['data'][cam_key]
        camera_image_path, _, cam_intrinsic = self.lyftd.get_sample_data(camera_token)

        boxes = self.get_boxes_from_token(sample_token)

        image_array = imread(camera_image_path)
        intrinsic = np.identity(3)

        ax.imshow(image_array)
        for pred_box in boxes:
            if pred_box.score > prob_threshold :
                box_in_camera_coord = transform_box_from_world_to_sensor_coordinates(pred_box, camera_token, self.lyftd)
                if box_in_camera_coord.center[2] > 0:
                    box_in_camera_coord.render(ax, view=cam_intrinsic, normalize=True, linewidth=2.0)

        ax.set_xlim([0, image_array.shape[1]])
        ax.set_ylim([image_array.shape[0], 0])

    def render_lidar_points(self, ax, sample_token, lidar_key='LIDAR_TOP', prob_threshold=0):

        lidar_top_token, lpc = self.get_lidar_points(lidar_key, sample_token)

        boxes = self.get_boxes_from_token(sample_token)

        for pred_box in boxes:
            if pred_box.score > prob_threshold:
                box_in_lidar_coord = transform_box_from_world_to_sensor_coordinates(pred_box, lidar_top_token,
                                                                                    self.lyftd)

                pts = lpc.points
                ax.scatter(pts[0, :], pts[1, :], s=0.05)
                ax.set_xlim([-50, 50])
                ax.set_ylim([-50, 50])
                view_mtx = np.eye(2)
                box_in_lidar_coord.render(ax, view=view_mtx)

    def get_lidar_points(self, lidar_key, sample_token):
        sample_record = self.lyftd.get('sample', sample_token)
        lidar_top_token = sample_record['data'][lidar_key]
        lidar_path = self.lyftd.get_sample_data_path(lidar_top_token)
        lpc = LidarPointCloud.from_file(lidar_path)
        return lidar_top_token, lpc

    def render_3d_lidar_points(self, sample_token, lidar_key='LIDAR_TOP', prob_threshold=0):

        lidar_token, lpc = self.get_lidar_points(lidar_key=lidar_key, sample_token=sample_token)

        fig = draw_lidar_simple(np.transpose(lpc.points))

        boxes = self.get_boxes_from_token(sample_token)

        box_pts = []
        for pred_box in boxes:
            if pred_box.score > prob_threshold:
                box_in_lidar_coord = transform_box_from_world_to_sensor_coordinates(pred_box, lidar_token,
                                                                                    self.lyftd)
                box_3d_pts = np.transpose(box_in_lidar_coord.corners())

                box_pts.append(box_3d_pts)

        draw_gt_boxes3d(box_pts, fig)

    def render_3d_lidar_points_to_camera_coordinates(self, sample_token, lidar_key="LIDAR_TOP",
                                                     cam_key="CAM_FRONT", prob_threshold=0):

        lidar_token, lpc = self.get_lidar_points(lidar_key=lidar_key, sample_token=sample_token)

        # Get camera coordiate calibration information
        sample_record = self.lyftd.get('sample', sample_token)
        camera_token = sample_record['data'][cam_key]

        camera_data = self.lyftd.get('sample_data', camera_token)
        lidar_record = self.lyftd.get('sample_data', lidar_token)

        lpc, _ = transform_pc_to_camera_coord(camera_data, lidar_record, lpc, self.lyftd)
        # Transform lidar points

        fig = draw_lidar_simple(np.transpose(lpc.points))

        boxes = self.get_boxes_from_token(sample_token)

        box_pts = []
        for pred_box in boxes:
            if pred_box.score > prob_threshold:
                box_in_lidar_coord = transform_box_from_world_to_sensor_coordinates(pred_box, camera_token,
                                                                                    self.lyftd)
                box_3d_pts = np.transpose(box_in_lidar_coord.corners())

                box_pts.append(box_3d_pts)

        draw_gt_boxes3d(box_pts, fig)

        # mlab.view(azimuth=270, elevation=150,
        #           focalpoint=[0, 0, 0], distance=62.0, figure=fig)
        return fig


def draw_lidar_simple(pc, color=None):
    ''' Draw lidar points. simplest set up. '''
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:, 2]
    # draw points
    mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color, color=None, mode='point', colormap='cool', scale_factor=1,
                  figure=fig)
    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
    # draw axis
    axes = np.array([
        [2., 0., 0., 0.],
        [0., 2., 0., 0.],
        [0., 0., 2., 0.],
    ], dtype=np.float64)
    mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)
    mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


def draw_lidar(pc, color=None, fig=None, bgcolor=(0, 0, 0), pts_scale=1, pts_mode='point', pts_color=None):
    ''' Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    '''
    if fig is None: fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[2, :]
    mlab.points3d(pc[0, :], pc[1, :], pc[2, :], color, color=pts_color, mode=pts_mode, colormap='gnuplot',
                  scale_factor=pts_scale, figure=fig)

    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)

    # draw axis
    axes = np.array([
        [2., 0., 0., 0.],
        [0., 2., 0., 0.],
        [0., 0., 2., 0.],
    ], dtype=np.float64)
    mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)

    # draw fov (todo: update to real sensor spec.)
    fov = np.array([  # 45 degree
        [20., 20., 0., 0.],
        [20., -20., 0., 0.],
    ], dtype=np.float64)

    mlab.plot3d([0, fov[0, 0]], [0, fov[0, 1]], [0, fov[0, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                figure=fig)
    mlab.plot3d([0, fov[1, 0]], [0, fov[1, 1]], [0, fov[1, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                figure=fig)

    # draw square region
    TOP_Y_MIN = -20
    TOP_Y_MAX = 20
    TOP_X_MIN = 0
    TOP_X_MAX = 40
    TOP_Z_MIN = -2.0
    TOP_Z_MAX = 0.4

    x1 = TOP_X_MIN
    x2 = TOP_X_MAX
    y1 = TOP_Y_MIN
    y2 = TOP_Y_MAX
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)

    # mlab.orientation_axes()
    mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


def draw_gt_boxes3d(gt_boxes3d, fig, color=(1, 1, 1), line_width=1, draw_text=True, text_scale=(1, 1, 1),
                    color_list=None):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    '''
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text: mlab.text3d(b[4, 0], b[4, 1], b[4, 2], '%d' % n, scale=text_scale, color=color, figure=fig)
        for k in range(0, 4):
            # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                        line_width=line_width, figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                        line_width=line_width, figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                        line_width=line_width, figure=fig)
    # mlab.show(1)
    # mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


if __name__ == '__main__':
    import pickle

    pfile = "/Users/kanhua/Downloads/3d-object-detection-for-autonomous-vehicles/artifacts/val_pc.pickle"

    with open(pfile, 'rb') as fp:
        item = pickle.load(fp)
        print(type(item))

    # point_cloud_3d = np.loadtxt('mayavi/kitti_sample_scan.txt')
    fig = draw_lidar_simple(item['pcl'][3])
    mlab.savefig('pc_view.jpg', figure=fig)
    input()

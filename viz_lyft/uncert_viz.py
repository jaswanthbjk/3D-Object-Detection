import os
import math
import numpy as np
import cv2
import kitti_util as utils
from PIL import Image
import mayavi.mlab as mlab


class kitti_object(object):
    '''Load and parse object data into a usable format.'''

    def __init__(self, root_dir, split='train'):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == 'train':
            self.num_samples = 240485
        elif split == 'testing':
            self.num_samples = 80161
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'image_2')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.lidar_dir = os.path.join(self.split_dir, 'velodyne')
        self.label_dir = os.path.join(self.split_dir, 'test_label_2')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert (idx < self.num_samples)
        img_filename = os.path.join(self.image_dir, '%06d.png' % (idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx):
        assert (idx < self.num_samples)
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin' % (idx))
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, idx):
        assert (idx < self.num_samples)
        calib_filename = os.path.join(self.calib_dir, '%06d.txt' % (idx))
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        assert (idx < self.num_samples and self.split == 'training')
        label_filename = os.path.join(self.label_dir, '%06d.txt' % (idx))
        return utils.read_label(label_filename)

    def get_depth_map(self, idx):
        pass

    def get_top_down(self, idx):
        pass


def show_image_with_boxes(img, objects, calib, show3d=True):
    """ Show image with 2D bounding boxes """
    img1 = np.copy(img)  # for 2d bbox
    img2 = np.copy(img)  # for 3d bbox
    color = (0, 255, 0)
    for obj in objects:
        if obj.type == 'DontCare':
            continue
        # if obj.t[2] < 25:
        #     continue
        if obj.entropy >= 0.8 or math.isnan(obj.entropy):
            color = (255, 0, 0)
        elif obj.entropy > 0.4 and obj.entropy < 0.8:
            color = (255, 255, 0)
        else:
            color = (0, 255, 0)

        # if obj.prob < 0.5 or math.isnan(obj.prob):
        #     color = (255, 0, 0)
        # elif obj.prob > 0.5 and obj.prob < 0.75:
        #     color = (255, 255, 0)
        # else:
        #     color = (0, 255, 0)
        print('color is {}'.format(color))
        cv2.rectangle(img1, (int(obj.xmin), int(obj.ymin)),
                      (int(obj.xmax), int(obj.ymax)), color, 2)
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color)
    Image.fromarray(img1).show()
    if show3d:
        Image.fromarray(img2).show()


def draw_lidar_simple(pc, color=None):
    ''' Draw lidar points. simplest set up. '''
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:, 2]
    # draw points
    mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color, color=None, mode='point', colormap='gnuplot', scale_factor=1,
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


def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
               (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def show_lidar_with_boxes(pc_velo, objects, calib,
                          img_fov=False, img_width=None, img_height=None):
    ''' Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) '''
    name2class = {'Car':0,'Pedestrian':1,'Cyclist':2}
    import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    # print(('All point num: ', pc_velo.shape[0]))
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 500))
    if img_fov:
        pc_velo = get_lidar_in_image_fov(pc_velo, calib, 0, 0, img_width, img_height)
        # print(('FOV point num: ', pc_velo.shape[0]))
    draw_lidar(pc_velo, fig=fig)

    for obj in objects:
        # print('{}:{}'.format(obj.type, obj.entropy))
        if obj.type == 'DontCare': continue
        # if obj.t[2] < 20:
        #     continue

        if obj.entropy >= 0.8 or math.isnan(obj.entropy):
            color = (1, 0, 0)
        elif obj.entropy > 0.4 and obj.entropy < 0.8:
            color = (1, 1, 0)
        else:
            color = (0, 1, 0)

        # if obj.prob < 0.5 or math.isnan(obj.prob):
        #     color = (1, 0, 0)
        # # elif obj.prob > 0.5 and obj.prob < 0.75:
        # #     color = (1, 1, 0)
        # else:
        #     color = (0, 1, 0)

        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        # Draw heading arrow
        ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]
        name = str(obj.type+str(', ')+str(round(obj.prob*100,2))+str(', ')+str(round(obj.entropy,2)))
        draw_gt_boxes3d([box3d_pts_3d_velo], name = name, color=color, fig=fig)
        mlab.plot3d([x1, x2], [y1, y2], [z1, z2], color=(0.5, 0.5, 0.5),
                    tube_radius=None, line_width=1, figure=fig)
    mlab.show(1)


dataset = kitti_object('/media/jarvis/CommonFiles/KITTI/')
data_idx = int(input())

# Load data from dataset
objects = dataset.get_label_objects(data_idx)
objects[0].print_object()
img = dataset.get_image(data_idx)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_height, img_width, img_channel = img.shape
pc_velo = dataset.get_lidar(data_idx)[:, 0:3]
calib = dataset.get_calibration(data_idx)

# Draw lidar in rect camera coord
show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height)

# Draw 2d and 3d boxes on image
print(' -------- 2D/3D bounding boxes in images --------')
show_image_with_boxes(img, objects, calib)
input()

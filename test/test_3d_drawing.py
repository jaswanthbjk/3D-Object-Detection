import pickle
import numpy as np
from dataset.prepare_lyft_data import extract_single_box, \
    parse_train_csv,level5data,extract_boxed_clouds,\
    get_train_data_sample_token_and_box,get_pc_in_image_fov,\
    extract_other_sensor_token,transform_box_from_world_to_sensor_coordinates
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud

from helpers.viz_util_for_lyft import draw_lidar_simple

def plot_demo_lyft_lidar():
    train_df = parse_train_csv()

    box, sample_data_token = extract_single_box(train_df, 0)

    ldp_file_path = level5data.get_sample_data_path(sample_data_token)
    lcdp = LidarPointCloud.from_file(ldp_file_path)

    draw_lidar_simple(lcdp.points[0:3,:])

    input()

def plot_cloud_in_box():
    pfile="/Users/kanhua/Downloads/3d-object-detection-for-autonomous-vehicles/artifacts/val_pc.pickle"

    with open(pfile,'rb') as fp:
        item=pickle.load(fp)
        print(type(item))

    #point_cloud_3d = np.loadtxt('mayavi/kitti_sample_scan.txt')
    fig = draw_lidar_simple(item['pcl'][3].T)
    #mlab.savefig('pc_view.jpg', figure=fig)
    input()

def plot_demo_random():
    pass

def plot_kitti_point_cloud():

    kitti_pc_file="/Users/kanhua/Downloads/3d-object-detection-for-autonomous-vehicles/artifacts/kitti_val_pc.pickle"

    with open(kitti_pc_file,'rb') as fp:
        item=pickle.load(fp)

    plot_pcl=item['pcl'][0]
    print(plot_pcl.shape)
    print(plot_pcl)
    draw_lidar_simple(plot_pcl.T)

    input()

def debug_bounding_box_in_cam_coord():
    train_df=parse_train_csv()
    data_idx=0
    sample_token, bounding_box = get_train_data_sample_token_and_box(data_idx, train_df)

    object_of_interest_name = ['car', 'pedestrian', 'cyclist']

    sample_record = level5data.get('sample', sample_token)

    lidar_data_token = sample_record['data']['LIDAR_TOP']

    w, l, h = bounding_box.wlh
    lwh = np.array([l, w, h])

    dummy_bounding_box = bounding_box.copy()
    mask, point_clouds_in_box, _, _, image = get_pc_in_image_fov(lidar_data_token, 'CAM_FRONT',
                                                                 bounding_box)
    assert dummy_bounding_box == bounding_box

    camera_token = extract_other_sensor_token('CAM_FRONT', lidar_data_token)
    bounding_box_sensor_coord = transform_box_from_world_to_sensor_coordinates(bounding_box, camera_token, )
    draw_lidar_simple(point_clouds_in_box)
    input()


#plot_demo_lyft_lidar()
#plot_cloud_in_box()

#plot_kitti_point_cloud()
debug_bounding_box_in_cam_coord()
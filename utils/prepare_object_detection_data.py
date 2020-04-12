from typing import List

import tensorflow as tf

from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix, \
    points_in_box, BoxVisibility
from prepare_lyft_data import level5data,get_paths
from utils import dataset_util
import numpy as np
import hashlib
import io
import pathlib
import PIL.Image

DATA_PATH, ARTIFACT_PATH, _ = get_paths()

object_classes = ['car', 'pedestrian', 'animal', 'other_vehicle', 'bus', 'motorcycle', 'truck', 'emergency_vehicle',
                  'bicycle']

object_idx_dict = {'car': 1, 'pedestrian': 2, 'animal': 3, 'other_vehicle': 4,
                   'bus': 5, 'motorcycle': 6, 'truck': 7, 'emergency_vehicle': 8,
                   'bicycle': 9}


def check_object_class_match():
    for idx, cat in enumerate(level5data.category):
        print(cat)
        if object_classes[idx] != cat['name']:
            print("{} does not match {}".format(object_classes[idx], cat['name']))


def get_2d_corners_from_projected_box_coordinates(projected_corners: np.ndarray):
    assert projected_corners.shape[0] == 3

    xmin = projected_corners[0, :].min()
    xmax = projected_corners[0, :].max()
    ymin = projected_corners[1, :].min()
    ymax = projected_corners[1, :].max()

    return xmin, xmax, ymin, ymax


def select_annotation_boxes(sample_token, lyftd: LyftDataset, box_vis_level: BoxVisibility = BoxVisibility.ALL,
                            camera_type=['CAM_FRONT', 'CAM_BACK','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK_LEFT']) -> (str, str, Box):
    """
    Select annotations that is a camera image defined by box_vis_level


    :param sample_token:
    :param box_vis_level:BoxVisbility.ALL or BoxVisibility.ANY
    :param camera_type: a list of camera that the token should be selected from
    :return: yield (sample_token,cam_token, Box)
    """
    sample_record = lyftd.get('sample', sample_token)

    cams = [key for key in sample_record["data"].keys() if "CAM" in key]
    cams = [cam for cam in cams if cam in camera_type]
    for cam in cams:

        # This step selects all the annotations in a camera image that matches box_vis_level
        cam_token = sample_record["data"][cam]
        image_filepath, boxes_in_sensor_coord, cam_intrinsic = lyftd.get_sample_data(
            cam_token, box_vis_level=box_vis_level, selected_anntokens=sample_record['anns']
        )

        sd_record = lyftd.get('sample_data', cam_token)
        img_width = sd_record['width']  # typically 1920
        img_height = sd_record['height']  # typically 1080

        CORNER_NUMBER = 4
        corner_list = np.empty((len(boxes_in_sensor_coord), CORNER_NUMBER))
        for idx, box_in_sensor_coord in enumerate(boxes_in_sensor_coord):
            # For perspective transformation, the normalization should set to be True
            box_corners_on_image = view_points(box_in_sensor_coord.corners(), view=cam_intrinsic, normalize=True)

            corners_2d = get_2d_corners_from_projected_box_coordinates(box_corners_on_image)
            corner_list[idx, :] = corners_2d

        yield image_filepath, cam_token, corner_list, boxes_in_sensor_coord, img_width, img_height


def create_tf_feature(image_file_path: pathlib.PosixPath,
                      camera_token: str,
                      corner_list: np.ndarray,
                      image_width: int,
                      image_height: int, boxes: List[Box]) -> tf.train.Example:
    box_feature_list = [(box.name, box.token, object_idx_dict[box.name]) for box in boxes]
    box_feature_list = list(map(list, zip(*box_feature_list)))

    BOX_NAME_INDEX = 0
    BOX_TOKEN_INDEX = 1
    BOX_NAME_ID_INDEX = 2
    classes_text_list = [s.encode('utf-8') for s in box_feature_list[BOX_NAME_INDEX]]
    anns_token_list = [s.encode('utf-8') for s in box_feature_list[BOX_TOKEN_INDEX]]

    with tf.gfile.GFile(image_file_path.as_posix(), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    file_basename = image_file_path.as_posix()

    feature_dict = {
        'image/height': dataset_util.int64_feature(image_height),
        'image/width': dataset_util.int64_feature(image_width),
        'image/filename': dataset_util.bytes_feature(
            file_basename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            camera_token.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(corner_list[:, 0] / float(image_width)),
        'image/object/bbox/xmax': dataset_util.float_list_feature(corner_list[:, 1] / float(image_width)),
        'image/object/bbox/ymin': dataset_util.float_list_feature(corner_list[:, 2] / float(image_height)),
        'image/object/bbox/ymax': dataset_util.float_list_feature(corner_list[:, 3] / float(image_height)),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text_list),
        'image/object/class/label': dataset_util.int64_list_feature(box_feature_list[2]),
        'image/object/class/anns_id': dataset_util.bytes_list_feature(anns_token_list)
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return example


def parse_protobuf_message(message: str):
    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/key/sha256':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/source_id':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/height':
            tf.FixedLenFeature((), tf.int64, default_value=1),
        'image/width':
            tf.FixedLenFeature((), tf.int64, default_value=1),
        # Image-level labels.
        'image/class/text':
            tf.VarLenFeature(tf.string),
        'image/class/label':
            tf.VarLenFeature(tf.int64),
        # Object boxes and classes.
        'image/object/bbox/xmin':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax':
            tf.VarLenFeature(tf.float32),
        'image/object/class/label':
            tf.VarLenFeature(tf.int64),
        'image/object/class/text':
            tf.VarLenFeature(tf.string),
        'image/object/area':
            tf.VarLenFeature(tf.float32),
        'image/object/is_crowd':
            tf.VarLenFeature(tf.int64),
        'image/object/difficult':
            tf.VarLenFeature(tf.int64),
        'image/object/group_of':
            tf.VarLenFeature(tf.int64),
        'image/object/weight':
            tf.VarLenFeature(tf.float32),

    }
    parsed_example = tf.io.parse_single_example(message, keys_to_features)
    filename = parsed_example['image/filename'].numpy().decode('UTF-8')
    xmin = parsed_example['image/object/bbox/xmin'].values.numpy()
    xmax = parsed_example['image/object/bbox/xmax'].values.numpy()
    ymin = parsed_example['image/object/bbox/ymin'].values.numpy()
    ymax = parsed_example['image/object/bbox/ymax'].values.numpy()

    print(ymax.shape)

    return filename, xmin, xmax, ymin, ymax

def write_data_to_files(entries_num):
    import contextlib2
    from dataset_tools import tf_record_creation_util
    from tqdm import tqdm
    from sklearn.model_selection import train_test_split

    default_train_file = os.path.join(DATA_PATH, "train.csv")

    df = pd.read_csv(default_train_file)

    def save_tf_record_file(output_file_base, num_shards,sel_indices):
        with contextlib2.ExitStack() as tf_record_close_stack:
            output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
                tf_record_close_stack, output_filebase, num_shards)

            for index in tqdm(sel_indices):
                sample_token = df.iloc[index, 0]
                for image_filepath, cam_token, corners, boxes, img_width, img_height in select_annotation_boxes(
                        sample_token,
                        level5data):
                    if len(boxes) > 0:
                        tf_example = create_tf_feature(image_file_path=image_filepath, camera_token=cam_token,
                                                       corner_list=corners, image_width=img_width,
                                                       image_height=img_height,
                                                       boxes=boxes)
                        output_shard_index = index % num_shards
                        output_tfrecords[output_shard_index].write(tf_example.SerializeToString())

    if entries_num is None:
        entries_num=df.shape[0]


    all_train_index=np.arange(entries_num)

    train_indices,val_indices=train_test_split(all_train_index,test_size=0.2)

    num_shards = 10
    type='train'
    output_filebase = os.path.join(ARTIFACT_PATH,'lyft_2d_{}.record'.format(type))
    save_tf_record_file(output_file_base=output_filebase,num_shards=num_shards,sel_indices=train_indices)

    num_shards= 10
    type='val'
    output_filebase = os.path.join(ARTIFACT_PATH,'lyft_2d_{}.record'.format(type))
    save_tf_record_file(output_file_base=output_filebase, num_shards=num_shards, sel_indices=val_indices)




if __name__ == "__main__":
    import pandas as pd
    import os
    from skimage.io import imread
    from vis_util import draw_bounding_boxes_on_image_array
    import matplotlib.pyplot as plt

    write_data_to_files(300)

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from dataset.prepare_lyft_data import get_paths
import pandas as pd
import os
import numpy as np

from dataset.prepare_object_detection_data import select_annotation_boxes,level5data,create_tf_feature,parse_protobuf_message

def test_run_through_sample_token():
    DATA_PATH, ARTIFACT_PATH, _ = get_paths()

    first_sample_token = '24b0962e44420e6322de3f25d9e4e5cc3c7a348ec00bfa69db21517e4ca92cc8'  # this is for test

    default_train_file = os.path.join(DATA_PATH, "train.csv")

    df = pd.read_csv(default_train_file)

    for id in df['Id']:

        for p, c in select_annotation_boxes(first_sample_token, level5data):
            print(p)
            print(c)

def test_write_and_read():
    from skimage.io import imread
    from helpers.vis_util import draw_bounding_boxes_on_image_array
    import matplotlib.pyplot as plt

    DATA_PATH, ARTIFACT_PATH, _ = get_paths()

    first_sample_token = '24b0962e44420e6322de3f25d9e4e5cc3c7a348ec00bfa69db21517e4ca92cc8'  # this is for test

    default_train_file = os.path.join(DATA_PATH, "train.csv")

    df = pd.read_csv(default_train_file)

    for image_filepath, cam_token, corners, boxes, img_width, img_height in select_annotation_boxes(first_sample_token,
                                                                                                    level5data):
        tf_example = create_tf_feature(image_file_path=image_filepath, camera_token=cam_token,
                                       corner_list=corners, image_width=img_width, image_height=img_height, boxes=boxes)
        example_message = tf_example.SerializeToString()
        filename, xmin, xmax, ymin, ymax = parse_protobuf_message(example_message)
        print(filename)

        print("image height: {}".format(img_height))
        print("image width: {}".format(img_width))
        image_array = imread(filename)

        box = np.vstack([ymin, xmin, ymax, xmax])
        box = np.transpose(box)

        draw_bounding_boxes_on_image_array(image_array, box)

        plt.figure()
        plt.imshow(image_array)
        plt.show()

if __name__=="__main__":

    test_write_and_read()
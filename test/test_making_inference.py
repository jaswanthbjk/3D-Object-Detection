"""
This file tests how the results are put together to be Box3D and the run evaluations


"""

import numpy as np
from helpers.parse_pointnet_output import read_frustum_pointnet_output_v2
from dataset.prepare_lyft_data_v2 import load_train_data
from lyft_dataset_sdk.eval.detection.mAP_evaluation import Box3D, get_ious
from lyft_dataset_sdk.utils.data_classes import Box
from test.test_data_loader import load_test_data
from typing import List
from absl import flags, app
import tqdm
from tqdm import trange
import pandas as pd
import math

FLAGS = flags.FLAGS
flags.DEFINE_bool("from_rgb_detection", False, "whether the frustum is generated from RGB detection")
flags.DEFINE_string("inference_file", None, "file output from test.py")
flags.DEFINE_string("pred_file", None, "output csv file name")
flags.DEFINE_string("data_name", 'train', "name of the data: train or test")


def box_to_box3D(box: Box, sample_token: str):
    box3d = Box3D(sample_token=sample_token, translation=box.center,
                  size=box.wlh, rotation=box.orientation.q, name=box.name, score=box.score)

    return box3d


from dataset.prepare_lyft_data import parse_string_to_box


class ScoreCalculator(object):

    def __init__(self, pred_csv_file: str, gt_csv_file: str):
        self.pred_df = pd.read_csv(pred_csv_file, index_col="Id")

        self.gt_df = pd.read_csv(gt_csv_file, index_col="Id")

    def get_box_string(self, df, sample_token: str):
        boxes_str = df.loc[sample_token, 'PredictionString']
        return boxes_str

    def calculate_single_entry(self, index: int):
        # Make boxes of gt_box
        sample_token_to_eval = self.pred_df.index[index]

        pred_box_string = self.get_box_string(self.pred_df, sample_token_to_eval)
        gt_box_string = self.get_box_string(self.gt_df, sample_token_to_eval)

        pred_3d_boxes = parse_string_to_box(pred_box_string, with_score=True, to_3dbox=True,
                                            sample_token=sample_token_to_eval)

        gt_3d_boxes = parse_string_to_box(gt_box_string, with_score=False, to_3dbox=True,
                                          sample_token=sample_token_to_eval)

        # calculate average
        ious = []
        for box in pred_3d_boxes:
            box_iou = get_ious(gt_3d_boxes, box)
            ious.append(box_iou)
        return np.array(ious)

    def calculate_mean_ious(self):

        all_max_ious=None
        print('Total number of samples are :{}'.format(self.pred_df.shape[0]))
        for i in trange(self.pred_df.shape[0]):
            all_ious = self.calculate_single_entry(i)
            max_ious = np.max(all_ious, axis=1)
            # max_ious_0 = np.max(all_ious, axis=0)
            # print(max_ious_0)
            # print(max_ious_1)
            if all_max_ious is not None:
                all_max_ious=np.concatenate((all_max_ious,max_ious.ravel()))
            else:
                all_max_ious=max_ious.ravel()

        return np.mean(all_max_ious)


def write_output_csv(pred_boxes: List[Box], sample_token_list, output_csv_file: str):
    with open(output_csv_file, 'w') as fp:
        fp.write("Id,PredictionString\n")
        prev_sample_token = None
        for idx, sample_token in enumerate(sample_token_list):
            # x, y, z, w, l, h, yaw, c
            score = pred_boxes[idx].score
            if math.isnan(score):
                continue
            x, y, z = pred_boxes[idx].center
            w, l, h = pred_boxes[idx].wlh
            yaw = pred_boxes[idx].orientation.angle * np.sign(pred_boxes[idx].orientation.axis[2])
            c = pred_boxes[idx].name

            pred_string = "{0} {1} {2} {3} {4} {5} {6} {7} {8} ".format(score, x, y, z, w, l, h, yaw, c)

            if sample_token != prev_sample_token:
                if prev_sample_token is not None:
                    fp.write("\n")
                fp.write(sample_token)
                fp.write(",")
            fp.write(pred_string)

            prev_sample_token = sample_token


# inference_pickle_file = "test_results.pickle"
# token_pickle_file = "/Users/kanhua/Dropbox/Programming/lyft-3d-main/artifact/lyft_val_token_from_rgb.pickle"
# pred_csv_file = "./test_pred.csv"
# FROM_RGB_DETECTION = True

def main(argv):
    inference_pickle_file = FLAGS.inference_file
    pred_csv_file = FLAGS.pred_file
    FROM_RGB_DETECTION = FLAGS.from_rgb_detection
    data_name = FLAGS.data_name
    if data_name == 'train':
        data = load_train_data()
    elif data_name == 'test':
        data = load_test_data()

    pred_boxes=[]
    sample_token_list=[]
    for box, sample_token in read_frustum_pointnet_output_v2(data,inference_pickle_file):
        pred_boxes.append(box)
        sample_token_list.append(sample_token)

    write_output_csv(pred_boxes, sample_token_list, pred_csv_file)

    # if not FROM_RGB_DETECTION:
    #     pred_boxes_3d = []
    #     gt_boxes_3d = []
    #
    #     for i in range(len(pred_boxes)):
    #         pred_boxes_3d.append(box_to_box3D(pred_boxes[i], sample_token_list[i]))
    #         gt_boxes_3d.append(box_to_box3D(gt_boxes[i], sample_token_list[i]))
    #
    #     for pbox in pred_boxes_3d:
    #         ious = get_ious(gt_boxes_3d, pbox)
    #         print(np.array(ious).max())




if __name__ == "__main__":
    app.run(main)

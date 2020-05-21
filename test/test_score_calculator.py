import pandas as pd

from test.test_making_inference import ScoreCalculator
from helpers.config_tool import get_paths
import os

data_path, _, _ = get_paths()

gt_file = './train.csv'

pred_file = './train_val_pred_kag_train.csv'

sc = ScoreCalculator(pred_csv_file=pred_file, gt_csv_file=gt_file)

# scores = sc.calculate_single_entry(1)

mean_score = sc.calculate_mean_ious()

# print(scores)

print(mean_score)

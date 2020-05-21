"""
Prepare test data

"""
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import pickle
from typing import Tuple, List
from PIL import Image

from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer

import matplotlib.pyplot as plt
from helpers.config_tool import get_test_data_path


def load_test_data():
    DATA_PATH = get_test_data_path()
    level5testdata = LyftDataset(data_path=DATA_PATH,json_path=os.path.join(DATA_PATH, 'data/'),verbose=True)
    print("number of scenes:", len(level5testdata.scene))
    return level5testdata

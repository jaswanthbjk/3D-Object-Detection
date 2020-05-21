import pickle
import numpy as np

from dataset.provider import FrustumDataset

from lyft_dataset_sdk.utils.geometry_utils import points_in_box

original_version_file = "./artifact/lyft_val.pickle"

new_version_file = "./artifact/lyft_val_2.pickle"

dataset_v1 = FrustumDataset(npoints=1024, split="val", overwritten_data_path=original_version_file)

dataset_v2 = FrustumDataset(npoints=1024, split="val", overwritten_data_path=new_version_file)

print(np.all(dataset_v1.input_list[0]== dataset_v2.input_list[0]))

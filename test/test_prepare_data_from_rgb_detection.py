import os
from dataset.prepare_lyft_data import prepare_frustum_data_from_scenes, level5data

# from test_data_loader import level5testdata
output_file = os.path.join("/dltraining/artifacts/lyft_val_from_rgb.pickle")
token_file = os.path.join("/dltraining/artifacts/lyft_val_token_from_rgb.pickle")
# prepare_frustum_data_from_traincsv(64, output_file)
prepare_frustum_data_from_scenes(100000, output_file, token_filename=token_file, scenes=range(151, 160),
                                 from_rgb_detection=True, lyftdf=level5data)

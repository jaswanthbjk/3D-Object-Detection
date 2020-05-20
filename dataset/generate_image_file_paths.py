from dataset.prepare_lyft_data_v2 import load_train_data, get_all_image_paths_in_single_scene
from test.test_data_loader import load_test_data
from helpers.config_tool import get_paths
import os

import pickle
from absl import app


class SceneImagePathSaver(object):
    def __init__(self, det_path, lyftd):
        self.det_path = det_path
        self.lyftd = lyftd

    def find_and_save_image_in_scene(self, scene_num):
        print("processing :{}".format(scene_num))
        all_images = [ip for ip in get_all_image_paths_in_single_scene(scene_number=scene_num, ldf=self.lyftd)]
        # TODO the file name should follow the type of dataset, train_scene_{} or test_scene_{}
        with open(os.path.join(self.det_path, "test_scene_{}_images.pickle".format(scene_num)), 'wb') as fp:
            pickle.dump(all_images, fp)


def main(argv):
    lyftd = load_test_data()
    data_path, artifacts_path, _ = get_paths()
    det_path = os.path.join(artifacts_path, "detection")
    
    print(det_path)

    scenes_to_process = range(0, 218, 1)
    sp = SceneImagePathSaver(det_path, lyftd)

    for scene_num in scenes_to_process:
        sp.find_and_save_image_in_scene(scene_num)

    # from multiprocessing import Pool
    # with Pool(processes=1) as p:
    #     p.map(sp.find_and_save_image_in_scene, scenes_to_process)


if __name__ == "__main__":
    app.run(main)

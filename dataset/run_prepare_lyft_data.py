import tensorflow as tf
from dataset.prepare_lyft_data_v2 import get_all_boxes_in_single_scene, load_train_data
from test.test_data_loader import load_test_data

from absl import flags, logging
from absl import app

from helpers.config_tool import get_paths
import os
import re

flags.DEFINE_list("scenes", 'all' , "scenes to be processed")
flags.DEFINE_string("data_type", "train", "file type")
flags.DEFINE_boolean("from_rgb", False, "whether from RGB detection")
flags.DEFINE_boolean("use_multisweep", False, "Use multisweep to collect Lidar points")
flags.DEFINE_boolean("use_detected_2d", False, "Load predetected 2D data")

FLAGS = flags.FLAGS


class SceneProcessor(object):

    def __init__(self, data_type, from_rgb_detection, use_multisweep, use_detected_2d=False):
        if data_type == "train":
            self.lyftd = load_train_data()
        elif data_type == "test":
            self.lyftd = load_test_data()
        else:
            raise ValueError("invalid data type. Valid dataset names are train or test")
        self.from_rgb_detection = from_rgb_detection
        self.data_type = data_type
        if self.from_rgb_detection:
            if use_detected_2d:
                from helpers.object_classifier import LoaderClassifier
                print('Loading Pre processed data')
                self.object_classifier = LoaderClassifier()
            else:
                from helpers.object_classifier import TLClassifier
                self.object_classifier = TLClassifier()
        else:
            self.object_classifier = None

        if self.from_rgb_detection:
            self.file_type = "rgb"
        else:
            self.file_type = "gt"

        self.use_multisweep = use_multisweep

    def process_one_scene(self, scene_num):
        print('Processing scene {}'.format(scene_num))
        _, artifact_path, _ = get_paths()
        augmented_data_path = os.path.join(artifact_path,'augmented_train_data')
        with tf.io.TFRecordWriter(
                os.path.join(augmented_data_path,
                             "scene_{0}_{1}_{2}.tfrec".format(scene_num, self.data_type, self.file_type))) as tfrw:
            for fp in get_all_boxes_in_single_scene(scene_number=scene_num, from_rgb_detection=self.from_rgb_detection,
                                                    ldf=self.lyftd,
                                                    use_multisweep=self.use_multisweep,
                                                    object_classifier=self.object_classifier):
                tfexample = fp.to_train_example()
                tfrw.write(tfexample.SerializeToString())

    def find_processed_scenes(self, data_path):

        pat = "scene_(\d+)_" + self.data_type + "_" + self.file_type + ".tfrec"

        processed_num_list = []
        for file in os.listdir(data_path):
            matched_object = re.match(pat, file)
            if matched_object is not None:
                processed_num_list.append(int(matched_object[1]))

        return processed_num_list


def list_all_files(data_dir=None, pat="scene_\d+_train.tfrec"):
    if data_dir is None:
        _, artifact_path, _ = get_paths()
        data_dir = artifact_path

    files = []
    for file in os.listdir(data_dir):
        files.append(os.path.join(data_dir, file))

    return files


def main(argv):
    from multiprocessing import Pool
    logging.set_verbosity(logging.INFO)

    sp = SceneProcessor(FLAGS.data_type, from_rgb_detection=FLAGS.from_rgb, use_multisweep=FLAGS.use_multisweep,
                        use_detected_2d=FLAGS.use_detected_2d)

    if "all" in FLAGS.scenes:
        scenes_to_process = range(218)
    elif "rest" in FLAGS.scenes:
        _, artifact_path, _ = get_paths()
        processed_scenes = sp.find_processed_scenes(artifact_path)
        scenes_to_process = []
        for i in range(218):
            if i not in processed_scenes:
                scenes_to_process.append(i)
    else:
        scenes_to_process = map(int, FLAGS.scenes)

    if FLAGS.from_rgb:  # it seems that object detector does not support parallel processes
        if FLAGS.use_detected_2d:
            logging.info("parallel processing RGB data:")
            # with Pool(processes=1) as p:
            #     p.map(sp.process_one_scene, scenes_to_process)
            for scene_num in scenes_to_process:
                sp.process_one_scene(scene_num)

        else:
            for s in scenes_to_process:
                # map(sp.process_one_scene, scenes_to_process)
                sp.process_one_scene(s)
    else:
        logging.info("Processing from GT data:")
        for scene_num in scenes_to_process:
            sp.process_one_scene(scene_num)


if __name__ == "__main__":
    app.run(main)

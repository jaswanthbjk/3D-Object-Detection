"""
Configure the main program.
This file is adapted from solcore5 (https://github.com/dalonsoa/solcore5)

"""

from configparser import ConfigParser
import os

config_file_dir = os.path.abspath(os.path.dirname(__file__))

default_config_file = "default_config.txt"

user_config_file = os.path.join(config_file_dir, 'user_config.txt')


def save_user_config(config_data):
    """ Saves the current user configuration

    :return: None
    """
    with open(user_config_file, 'w') as fp:
        config_data.write(fp)


def generate_default_setting():
    default_config_data = ConfigParser()
    default_config_data.read(os.path.join(config_file_dir, default_config_file))

    save_user_config(default_config_data)


def load_user_config_data():
    if not os.path.exists(user_config_file):
        generate_default_setting()

    config_data = ConfigParser()
    config_data.read(os.path.join(config_file_dir, user_config_file))

    return config_data


def set_paths(data_path, test_data_path, artifact_path, model_checkpoint, object_detection_model_path):
    config_data = load_user_config_data()
    config_data['path_config']['model_checkpoint'] = model_checkpoint
    config_data['path_config']['artifact_path'] = artifact_path
    config_data['path_config']['data_path'] = data_path
    config_data['path_config']['test_data_path'] = test_data_path
    config_data['path_config']['object_detection_model_path'] = object_detection_model_path

    save_user_config(config_data)


def get_paths() -> (str, str, str):
    """

    :return: data_path,artifact_path,model_checkpoint
    """
    config_data = load_user_config_data()
    data_path = '/media/jarvis/CommonFiles/v1_02/v1.02-train'
    artifact_path = '/media/jarvis/CommonFiles/v1_02/v1.02-train/Artifacts'
    model_path = './log_v1/'

    return data_path, artifact_path, model_path


def get_test_data_path():
    config_data = load_user_config_data()
    test_data_path = '/scratch/jbandl2s/Lyft_dataset/Kaggle_data/Lyft_test'

    return test_data_path


def get_object_detection_model_path():
    config_data = load_user_config_data()

    return config_data['path_config']['object_detection_model_path']

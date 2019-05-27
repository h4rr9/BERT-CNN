import json
import os
import time

from dotmap import DotMap


def get_config_from_json(json_file):

    with open(json_file, "r") as config_file:
        config_dict = json.load(config_file)

    config = DotMap(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)

    config.callbacks.tensorboard_log_dir = os.path.join("experiments", time.strftime(
        "%Y-%m-%d", time.localtime()), config.exp.name, "logs")
    config.callbacks.checkpoint_dir = os.path.join("experiments", time.strftime(
        "%Y-%m-%d", time.localtime()), config.exp.name, "checkpoints")

    return config

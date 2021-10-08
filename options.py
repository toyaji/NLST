import argparse
import sys
import yaml
from easydict import EasyDict
from datetime import datetime


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
        config = EasyDict(config)
    return config

def load_config_from_args():
    args = argparse.ArgumentParser()
    args.add_argument("-c", "--config", required=True, help="You can see the sample yaml template in /config folder.")
    args.add_argument("-n", "--name", type=str, help="You can put the name for the experiment, this will be used for log file name.")
    args.add_argument("-p", "--patch", type=int, help="Patch size for data loader crop and model generation.")
    args.add_argument("--chop_size", type=int, help="For test forward, we need to chop the input according to its memory capability.")
    args.add_argument("--test_only", action='store_true', help='set this option to test the model')
    args.add_argument("--save_test_img", action='store_true', help='set this option to test the model')
    args = args.parse_args(sys.argv[1:])
    
    config = load_config(args.config)
    config.log.name = args.name
    config.log.version = 'log_' + datetime.now().strftime("%y%m%d%H%M")
    config.dataset.args.patch_size = args.patch
    config.dataset.test_only = args.test_only
    config.dataset.save_test_img = args.save_test_img
    config.model.chop_size = args.chop_size
    return config
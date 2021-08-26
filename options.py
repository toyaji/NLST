import argparse
import sys
from pathlib import Path
from pprint import pformat

import yaml
from easydict import EasyDict


def load_config(config_file, write_log=True):
    with open(config_file, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
        config = EasyDict(config)

    return config

def load_config_from_args():
    args = argparse.ArgumentParser()
    args.add_argument("config")
    args = args.parse_args(sys.argv[1:])

    return load_config(args.config)
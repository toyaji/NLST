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

def get_model_args(config):
    model = config.model.net
    if model == "HAN":
        config.model.n_resgroups = 10
        config.model.n_resblocks = 20
        config.model.n_feats = 64
        config.model.reduction = 16
        config.model.n_colors = 3
        config.model.res_scale = 1

    elif model == "SCAN":
        config.model.n_resgroups = 8
        config.model.n_resblocks = 12
        config.model.n_feats = 64
        config.model.channels = [128, 256, 512, 512]
        config.model.reduction = [2, 4, 8, 8]
        config.model.n_colors = 3
        config.model.res_scale = 1
        config.model.extractor_ver = 'vgg19'
        config.model.extractor_grad = True
        config.model.extractor_train = True

    elif model == "NONSCAN":
        config.model.img_size = config.dataset.args.patch_size
        config.model.n_strablocks = 15
        config.model.n_stratum = 5
        config.model.n_feats = 128
        config.model.work_dim = 64
        config.model.reduction = [2, 4, 8, 16]
        config.model.n_colors = 3
        config.model.res_scale = 1
        config.model.concat = True

    elif model == "NLST":
        config.model.img_size = config.dataset.args.patch_size
        config.model.n_strablocks = 1
        config.model.n_stratum = 1
        config.model.n_feats = 64
        config.model.work_dim = 32
        config.model.reduction = [1, 2, 4, 8]
        config.model.n_colors = 3
        config.model.res_scale = 1
        config.model.concat = True

    elif model == "CSNLN":
        config.model.depth = 12
        config.model.n_resblocks = 16
        config.model.n_feats = 128
        config.model.n_colors = 3

    elif model == "SwinIR":
          config.model.img_size = config.dataset.args.patch_size
          config.model.window_size = 8
          config.model.embed_dim = 180
          config.model.depths = [6, 6, 6, 6, 6]
          config.model.num_heads = [6, 6, 6, 6, 6]
          config.model.n_colors = 3
          config.model.mlp_ratio = 2
          config.model.upsampler = 'pixelshuffle'
          config.model.resi_connection = '1conv'

    return config

def load_config_from_args():
    # thouh we have template for each models, still you can add any options through belowing code
    args = argparse.ArgumentParser()
    args.add_argument("-c", "--config", type=str, help="You can see the sample yaml template in /config folder.")
    args.add_argument("-m", "--model", type=str, help="Model for training")
    args.add_argument("-n", "--name", type=str, help="You can put the name for the experiment, this will be used for log file name.")
    args.add_argument("-p", "--patch", type=int, help="Patch size for data loader crop and model generation.")
    args.add_argument("-b", "--batch", type=int, help="Batch size for data laoder.")
    args.add_argument("-s", "--scale", type=int, help="Scale factor.")
    args.add_argument("-w", "--workers", type=int, help="Number of worker for dataload")
    args.add_argument("--chop_size", type=int, help="For test forward, we need to chop the input according to its memory capability.")
    args.add_argument("--test_only", action='store_true', help='set this option to test the model')
    args.add_argument("--save_imgs", action='store_true', help='set this option to test the model')
    args = args.parse_args(sys.argv[1:])
    
    if args.config is None:
        args.config = "config/base_template.yaml"

    config = load_config(args.config)
    config.log.name = args.name
    config.log.version = 'log_' + datetime.now().strftime("%y%m%d%H%M")

    if args.model is not None:
        config.model.net = args.model

    if args.patch is not None:
        config.dataset.args.patch_size = args.patch

    if args.batch is not None:
        config.dataset.batch_size = args.batch

    if args.workers is not None:
        config.dataset.num_workers = args.workers
    
    config.test_only = args.test_only

    if args.save_imgs:
        config.dataset.save_test_img = True

    config.model.chop_size = args.chop_size

    config = get_model_args(config)

    # rgb range set copy to model param
    config.model.rgb_range = config.dataset.args.rgb_range
    config.model.scale = config.dataset.args.scale
    return config
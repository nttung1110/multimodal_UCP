import os.path as osp
import os
import pdb

def mkdir(path):
    os.makedirs(path, exist_ok=True)

def setup_dir(cfg):
    # setup output dir
    next_dir = osp.join(cfg.output_dir, cfg.data.path_save_json)
    mkdir(next_dir)
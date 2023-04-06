from src.utils import mkdir
from .base_folder import BasefolderLoader

import os
import os.path as osp

class VideofolderLoader(BasefolderLoader):
    def __init__(self, cfg):
        super(VideofolderLoader, self).__init__(cfg)

    def get_list_item(self):
        item_list = []

        for file_name in os.listdir(self.cfg.data.path_input_folder):
            f_name_no_ext = file_name.split('.')[0]
            
            file_path_in = osp.join(self.cfg.data.path_input_folder, file_name)
            file_path_out = osp.join(self.write_path_json, f_name_no_ext+'.json')

            start_second = 0 # for starting the segment of video in a large video
            item_list.append((file_path_in, file_path_out, start_second)) 

        return item_list
    
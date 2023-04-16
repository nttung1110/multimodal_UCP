from src.utils import mkdir
from .base_folder import BasefolderLoader

import os
import os.path as osp
import pandas as pd
import pdb

class VideoCSVLoader(BasefolderLoader):
    def __init__(self, cfg):
        super(VideoCSVLoader, self).__init__(cfg)

    def get_list_item(self):
        item_list = []
        # pdb.set_trace()
        df = pd.read_csv(self.cfg.data.path_input_folder)
        
        for idx, row in df.iterrows():
            f_name_no_ext = row['segment_id']
            
            file_path_in = row['file_path']
            file_path_out = osp.join(self.write_path_json, f_name_no_ext+'.json')

            start_second = 0 
            item_list.append((file_path_in, file_path_out, start_second)) 

        return item_list
    
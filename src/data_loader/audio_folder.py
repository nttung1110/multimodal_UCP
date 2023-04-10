# from .base_folder import BaseFolderLoader
# from src.utils import mkdir

# import os
# import os.path as osp

# class AudioFolder(BaseFolderLoader):
#     def __init__(self, cfg):
#         super(AudioFolder, self).__init__(cfg)
#         self.write_path_json = osp.join(cfg.output_dir,
#                                         cfg.data.path_audio_save_json)

#     def get_list_item(self):
#         item_list = []

#         for file_name in os.listdir(self.cfg.data.path_audio_folder):
#             f_name_no_ext = file_name.split('.')[0]
            
#             file_path_in = osp.join(self.cfg.data.path_audio_folder, file_name)
#             file_path_out = osp.join(self.write_path_json, f_name_no_ext+'.json')

#             item_list.append((file_path_in, file_path_out))

#         return item_list
    
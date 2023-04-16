import cv2
import bbox_visualizer as bbv
import os.path as osp
import numpy as np
import pdb
from tqdm import tqdm
from pathlib import Path


class BaseTracker():
    # base class for tracker
    def __init__(self, config):
        self.config = config
        self.number_of_missing_frames = 0

    def _debug_visualize_track_emotion(self, frames_list, f_p_in, fps, w, h):
        # from frame list and all track, drawing face bbox
        # FOR VISUALIZING ONLY, draw all face track box
        print('=======Write video debugging=======')

        video_name = f_p_in.split('/')[-1]
        debug_vid_path = osp.join(self.config.folder_debug_tracking, video_name)
        Path(self.config.folder_debug_tracking).mkdir(parents=True, exist_ok=True)
        
        # init video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output = cv2.VideoWriter(debug_vid_path, fourcc, fps, (w, h), True)

        # construct frame writing annotation information dict
        
        frame_write = {}
        for idx_track, cur_track_info in self.all_tracks.items():
            cur_track_emotion = self.all_emotion_category_tracks[idx_track]
            # cur_feat = self.all_es_feat_tracks[idx_track]

            if len(cur_track_emotion) != len(cur_track_info['bbox']):
                assert RuntimeError # misalign emotion prediction and box, check again
            
            for idx_bbox_track in range(len(cur_track_info['bbox'])):
                where_face = cur_track_info['bbox'][idx_bbox_track]
                which_frame = cur_track_info['frames_appear'][idx_bbox_track]
                
                if which_frame not in frame_write:
                    frame_write[which_frame] = []

                # annotation text for box writing in frame
                emotion_cat = cur_track_emotion[idx_bbox_track]
                text_box = str(f'ID:{idx_track}->{emotion_cat}')

                # box writing in frame
                box = where_face
                frame_write[which_frame].append((box, text_box))
                
        for idx, frame in tqdm(enumerate(frames_list)):
            annot_frame = frame
            annot_frame = cv2.cvtColor(annot_frame, cv2.COLOR_RGB2BGR) 
            if idx in frame_write:
                # there are box and text need to be written in this frame
                for (each_box, each_text) in frame_write[idx]:
                    annot_frame = bbv.draw_rectangle(annot_frame, each_box)
                    annot_frame = bbv.add_label(annot_frame, each_text, each_box)

            output.write(annot_frame)

            if idx >= self.config.max_frame_debug:
                break

        output.release()

    def _create_new_track(self, es_feature, emotion_cat, 
                        bbox, track_id, frames_appear):
        # Initialize a new track
        new_box = {"bbox": [bbox], "frames_appear": [frames_appear]}
        self.all_tracks[track_id] = new_box

        new_ec_array_track = np.array([emotion_cat])
        self.all_emotion_category_tracks[track_id] = new_ec_array_track
    
        new_es_array_track = np.array([es_feature])
        self.all_es_feat_tracks[track_id] = new_es_array_track
        
        new_start_end_offset_track = [frames_appear, frames_appear]
        self.all_start_end_offset_track[track_id] = new_start_end_offset_track

    def _update_old_track(self, es_feature, emotion_cat, 
                        bbox, track_id, frames_appear):
        
        ## TODO: interpolate features for the missing frames?
        # Update existing track
        self.all_tracks[track_id]['bbox'].append(bbox)
        self.all_tracks[track_id]['frames_appear'].append(frames_appear)
        
        time_interpolate = frames_appear - self.all_tracks[track_id]['frames_appear'][-2] - 1

        if time_interpolate > 0:
            self.number_of_missing_frames += time_interpolate
            old_rep_track = self.all_es_feat_tracks[track_id][-1].tolist()
            self.all_es_feat_tracks[track_id] = np.append(self.all_es_feat_tracks[track_id], [old_rep_track]*time_interpolate, axis=0)


        new_es_array_track = np.array([es_feature])
        self.all_es_feat_tracks[track_id] = np.concatenate([self.all_es_feat_tracks[track_id], new_es_array_track])
        self.all_emotion_category_tracks[track_id] = np.concatenate([self.all_emotion_category_tracks[track_id], [emotion_cat]])
        self.all_start_end_offset_track[track_id][-1] = frames_appear


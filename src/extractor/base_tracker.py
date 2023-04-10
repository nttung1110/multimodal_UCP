import cv2
import bbox_visualizer as bbv
import os.path as osp
import numpy as np

from tqdm import tqdm


class BaseTracker():
    # base class for tracker
    def __init__(self, config):
        self.config = config

    def _debug_visualize_track_emotion(self, frames_list, f_p_in, fps, w, h):
        # from frame list and all track, drawing face bbox
        # FOR VISUALIZING ONLY, draw all face track box
        print('=======Write video debugging=======')

        video_name = f_p_in.split('/')[-1]
        debug_vid_path = osp.join(self.config.folder_debug_tracking, video_name)
        
        # init video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output = cv2.VideoWriter(debug_vid_path, fourcc, fps, (w, h), True)

        # construct frame writing annotation information dict
        frame_write = {}
        for idx_track, _ in enumerate(self.all_tracks):
            cur_track_info = self.all_tracks[idx_track]
            cur_track_emotion = self.all_emotion_category_tracks[idx_track]
            cur_feat = self.all_es_feat_tracks[idx_track]

            if len(cur_track_emotion) != len(cur_track_info['bbox']):
                assert RuntimeError # misalign emotion prediction and box, check again
            
            for idx_bbox_track in range(len(cur_track_info['bbox'])):
                where_face = cur_track_info['bbox'][idx_bbox_track]
                which_frame = cur_track_info['frames_appear'][idx_bbox_track]
                
                if which_frame not in frame_write:
                    frame_write[which_frame] = []

                # annotation text for box writing in frame
                id = cur_track_info['id']
                emotion_cat = cur_track_emotion[idx_bbox_track]
                text_box = str(f'ID:{id}->{emotion_cat}')

                # box writing in frame
                box = where_face
                frame_write[which_frame].append((box, text_box))

        for idx, frame in tqdm(enumerate(frames_list)):
            annot_frame = frame
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
        new_track = {"bbox": [bbox], "id": track_id, "frames_appear": [frames_appear]}

        new_es_array_track = np.array([es_feature])
        new_start_end_offset_track = [frames_appear, frames_appear]  # [start, end]
        new_ec_array_track = np.array([emotion_cat])

        self.all_tracks.append(new_track)
        self.all_emotion_category_tracks.append(new_ec_array_track)
        self.all_es_feat_tracks.append(new_es_array_track)
        self.all_start_end_offset_track.append(new_start_end_offset_track)

    def _update_old_track(self, es_feature, emotion_cat, 
                        bbox, which_track_id, frames_appear):
        # Update existing track
        self.all_tracks[which_track_id]['bbox'].append(bbox)
        self.all_tracks[which_track_id]['frames_appear'].append(frames_appear)

        time_interpolate = frames_appear - self.all_tracks[which_track_id]['frames_appear'][-2] - 1
        
        if time_interpolate > 0:
            old_rep_track = self.all_es_feat_tracks[which_track_id][-1].tolist()
            self.all_es_feat_tracks[which_track_id] = np.append(self.all_es_feat_tracks[which_track_id], [old_rep_track] * time_interpolate, axis=0)
        
        self.all_es_feat_tracks[which_track_id] = np.append(self.all_es_feat_tracks[which_track_id], [es_feature], axis=0)  # add more feature for this track
        self.all_start_end_offset_track[which_track_id][-1] = frames_appear  # change index frame
        self.all_emotion_category_tracks[which_track_id] = np.append(self.all_emotion_category_tracks[which_track_id], [emotion_cat], 0)

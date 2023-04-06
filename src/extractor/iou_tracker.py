from src.utils import Config
from tqdm import tqdm

import torch
import cv2
import numpy as np

class IoUTracker():
    def __init__(self, config: Config):
        self.config = config

        # init model
        self._init_model()

    def _init_model(self):

        print('=========Initializing IOU Face Tracker Model=========')
        

    def _cal_iou(self, bbox1, bbox2):
        """
        Calculates the intersection-over-union of two bounding boxes.
            Args:
                bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
                bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
            Returns:
                int: intersection-over-onion of bbox1, bbox2
        """
        bbox1 = [float(x) for x in bbox1]
        bbox2 = [float(x) for x in bbox2]

        (x0_1, y0_1, x1_1, y1_1) = bbox1
        (x0_2, y0_2, x1_2, y1_2) = bbox2

        # get the overlap rectangle
        overlap_x0 = max(x0_1, x0_2)
        overlap_y0 = max(y0_1, y0_2)
        overlap_x1 = min(x1_1, x1_2)
        overlap_y1 = min(y1_1, y1_2)

        # check if there is an overlap
        if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            return 0

        # if yes, calculate the ratio of the overlap to each ROI size and the unified size
        size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
        size_union = size_1 + size_2 - size_intersection

        return size_intersection / size_union
    
    def _match_and_update_track(self, box):
        best_match_track_id = None
        best_match_track_score = 0
        for idx, each_active_tracks in enumerate(self.all_tracks):
            if idx in self.mark_old_track_idx:
                # ignore inactive track
                continue
            latest_track_box = each_active_tracks['bbox'][-1]

            iou_score = self._cal_iou(latest_track_box, box)
            
            if iou_score > best_match_track_score and iou_score > self.config.threshold_iou_min_track:
                best_match_track_id = idx
                best_match_track_score = iou_score

        return best_match_track_id, best_match_track_score
    
    def _maintain_track_status(self, idx_frame):
        for idx, each_active_tracks in enumerate(self.all_tracks):
            old_idx_frame = each_active_tracks['frames_appear'][-1]

            if idx_frame - old_idx_frame > self.config.threshold_dying_track_len:
                # this is the inactive track, mark it
                self.mark_old_track_idx.append(idx)
    
    def _init_list_record(self):
        self.all_tracks = []
        self.mark_old_track_idx = []

        self.all_emotion_category_tracks = []
        self.all_es_feat_tracks = []
        self.all_start_end_offset_track = []

    
    def _creat_or_update(self, bboxes, emot_extractor, idx_frame, frame):
        # create if box is for new track, update if box belong to old track
        for _, bbox in enumerate(bboxes):
            box = bbox.astype(int)            
            # ====================
            # Stage 2.0: Extracting ES features from facial image
            
            _, es_feature, emotion_cat = emot_extractor.run(frame, box)

            # =====================
            # Stage 2.1: Finding to which track this es_feature belongs to based on iou
        
            # finding which track this box belongs to
            best_match_track_id, _ = self._match_and_update_track(box)
            if best_match_track_id is None:
                # there is no active track currently, then this will initialize a new track
                new_track = {"bbox": [box], "id": len(self.all_tracks), "frames_appear": [idx_frame]}


                # also create new np array representing for new track here
                new_es_array_track = np.array([es_feature])
                new_start_end_offset_track = [idx_frame, idx_frame] #[start, end]
                new_ec_array_track = np.array([emotion_cat])

                self.all_tracks.append(new_track)
                self.all_emotion_category_tracks.append(new_ec_array_track)
                self.all_es_feat_tracks.append(new_es_array_track)
                self.all_start_end_offset_track.append(new_start_end_offset_track)
            else:
                # update track
                self.all_tracks[best_match_track_id]['bbox'].append(box)
                self.all_tracks[best_match_track_id]['frames_appear'].append(idx_frame)

                # update all_list

                ### interpolate first

                time_interpolate = idx_frame - self.all_tracks[best_match_track_id]['frames_appear'][-2] - 1

                if time_interpolate > 0:
                    old_rep_track = self.all_es_feat_tracks[best_match_track_id][-1].tolist()
                    self.all_es_feat_tracks[best_match_track_id] = np.append(self.all_es_feat_tracks[best_match_track_id], [old_rep_track]*time_interpolate, axis=0)

                ### then do update
                self.all_es_feat_tracks[best_match_track_id] = np.append(self.all_es_feat_tracks[best_match_track_id], [es_feature], axis=0) # add more feature for this track
                self.all_start_end_offset_track[best_match_track_id][-1] = idx_frame # change index frame


    def _filter_tracks(self):
        # filter those tracks having length smaller than a number
        all_es_feat_tracks_filter = []
        all_start_end_offset_track_filter = []
        all_emotion_category_tracks_filter = []

        for es_feat_track, se_track, ec_track in zip(self.all_es_feat_tracks, 
                                                     self.all_start_end_offset_track, 
                                                     self.all_emotion_category_tracks):
            length = es_feat_track.shape[0]
            if length >= self.config.len_face_tracks:
                all_es_feat_tracks_filter.append(es_feat_track)
                all_start_end_offset_track_filter.append(se_track)
                all_emotion_category_tracks_filter.append(ec_track)

        return all_es_feat_tracks_filter, all_start_end_offset_track_filter, all_emotion_category_tracks_filter

    def run(self, video, face_detector, emot_extractor):
        # setup a new set of record list everytime processing a video
        self._init_list_record()
        frames_list = list(video.iter_frames())

        for idx_frame, frame in tqdm(enumerate(frames_list), total=len(frames_list)):
            if idx_frame % self.config.skip_frame != 0:
                continue        
                
            # detect faces
            bboxes, _ = face_detector.run(frame)

            if bboxes is None:
                continue

            # Stage 1: Maintaining track status to kill inactive track
            self._maintain_track_status(idx_frame)

            # Stage 2: Assign new boxes to currently active tracks or create a new track if there are no active tracks
            self._creat_or_update(bboxes, emot_extractor, idx_frame, frame)

            # debug mode
            if self.config.is_debug:
                if idx_frame > 400:
                    break
        
        # get final result
        all_es, all_se_offset, all_emot_cat = self._filter_tracks()

        return all_es, all_se_offset, all_emot_cat


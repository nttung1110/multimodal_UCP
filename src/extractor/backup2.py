# # TUNGCODE for base_tracker.py


# import cv2
# import bbox_visualizer as bbv
# import os.path as osp
# import numpy as np

# from tqdm import tqdm


# class BaseTracker():
#     # base class for tracker
#     def __init__(self, config):
#         self.config = config

#     def _debug_visualize_track_emotion(self, frames_list, f_p_in, fps, w, h):
#         # from frame list and all track, drawing face bbox
#         # FOR VISUALIZING ONLY, draw all face track box
#         print('=======Write video debugging=======')

#         video_name = f_p_in.split('/')[-1]
#         debug_vid_path = osp.join(self.config.folder_debug_tracking, video_name)
        
#         # init video writer
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         output = cv2.VideoWriter(debug_vid_path, fourcc, fps, (w, h), True)

#         # construct frame writing annotation information dict
#         frame_write = {}
#         for idx_track, _ in enumerate(self.all_tracks):
#             cur_track_info = self.all_tracks[idx_track]
#             cur_track_emotion = self.all_emotion_category_tracks[idx_track]
#             cur_feat = self.all_es_feat_tracks[idx_track]

#             if len(cur_track_emotion) != len(cur_track_info['bbox']):
#                 assert RuntimeError # misalign emotion prediction and box, check again
            
#             for idx_bbox_track in range(len(cur_track_info['bbox'])):
#                 where_face = cur_track_info['bbox'][idx_bbox_track]
#                 which_frame = cur_track_info['frames_appear'][idx_bbox_track]
                
#                 if which_frame not in frame_write:
#                     frame_write[which_frame] = []

#                 # annotation text for box writing in frame
#                 id = cur_track_info['id']
#                 emotion_cat = cur_track_emotion[idx_bbox_track]
#                 text_box = str(f'ID:{id}->{emotion_cat}')

#                 # box writing in frame
#                 box = where_face
#                 frame_write[which_frame].append((box, text_box))

#         for idx, frame in tqdm(enumerate(frames_list)):
#             annot_frame = frame
#             if idx in frame_write:
#                 # there are box and text need to be written in this frame
#                 for (each_box, each_text) in frame_write[idx]:
#                     annot_frame = bbv.draw_rectangle(annot_frame, each_box)
#                     annot_frame = bbv.add_label(annot_frame, each_text, each_box)
#                     annot_frame = = cv2.cvtColor(annot_frame, cv2.COLOR_RGB2BGR) 

#             output.write(annot_frame)

#             if idx >= self.config.max_frame_debug:
#                 break

#         output.release()

#     def _create_new_track(self, es_feature, emotion_cat, 
#                         bbox, track_id, frames_appear):
#         # Initialize a new track
#         new_track = {"bbox": [bbox], "id": track_id, "frames_appear": [frames_appear]}

#         new_es_array_track = np.array([es_feature])
#         new_start_end_offset_track = [frames_appear, frames_appear]  # [start, end]
#         new_ec_array_track = np.array([emotion_cat])

#         self.all_tracks.append(new_track)
#         self.all_emotion_category_tracks.append(new_ec_array_track)
#         self.all_es_feat_tracks.append(new_es_array_track)
#         self.all_start_end_offset_track.append(new_start_end_offset_track)

#     def _update_old_track(self, es_feature, emotion_cat, 
#                         bbox, which_track_id, frames_appear):
#         # Update existing track
#         self.all_tracks[which_track_id]['bbox'].append(bbox)
#         self.all_tracks[which_track_id]['frames_appear'].append(frames_appear)

#         time_interpolate = frames_appear - self.all_tracks[which_track_id]['frames_appear'][-2] - 1
        
#         if time_interpolate > 0:
#             old_rep_track = self.all_es_feat_tracks[which_track_id][-1].tolist()
#             self.all_es_feat_tracks[which_track_id] = np.append(self.all_es_feat_tracks[which_track_id], [old_rep_track] * time_interpolate, axis=0)
        
#         self.all_es_feat_tracks[which_track_id] = np.append(self.all_es_feat_tracks[which_track_id], [es_feature], axis=0)  # add more feature for this track
#         self.all_start_end_offset_track[which_track_id][-1] = frames_appear  # change index frame
#         self.all_emotion_category_tracks[which_track_id] = np.append(self.all_emotion_category_tracks[which_track_id], [emotion_cat], 0)
        
    





# # TUNGCODE for deep_sort_tracker.py
# from src.utils import Config
# from tqdm import tqdm
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultTrainer
# from fastreid.utils.checkpoint import Checkpointer
# from scipy.optimize import linear_sum_assignment as linear_assignment

# import pdb
# import torch
# import torchvision.transforms as transforms
# import numpy as np
# import cv2
# import logging
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import scipy.linalg
# INFTY_COST = 1e+5

# from .base_tracker import BaseTracker


# class DeepSortTracker(BaseTracker):
#     def __init__(self, cfg):
#         super(DeepSortTracker, self).__init__(cfg)
#         self._init_model(cfg)

#     def _init_model(self, cfg):

#         print('=========Initializing Deep Sort Tracker Model=========')
#         self.min_confidence = cfg.min_confidence
#         self.nms_max_overlap = cfg.nms_max_overlap

#         self.extractor = Extractor(cfg.model_path, use_cuda=cfg.use_cuda)
#         max_cosine_distance = cfg.max_dist
#         metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, cfg.nn_budget)
#         self.tracker = Tracker(metric, max_iou_distance=cfg.max_iou_distance, max_age=cfg.max_age, n_init=cfg.n_init)

#     def update(self, bbox_xywh, confidences, ori_img):
#         self.height, self.width = ori_img.shape[:2]
#         # generate detections
#         features = self._get_features(bbox_xywh, ori_img)
#         bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
#         detections = [Detection(bbox_tlwh[i], conf, features[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]

#         # run on non-maximum supression
#         boxes = np.array([d.tlwh for d in detections])
#         scores = np.array([d.confidence for d in detections])
#         indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
#         detections = [detections[i] for i in indices]

#         # update tracker
#         self.tracker.predict()
#         self.tracker.update(detections)

#         # output bbox identities
#         outputs = []

#         # pdb.set_trace()
#         for track in self.tracker.tracks:
#             if not track.is_confirmed() or track.time_since_update > 1:
#                 continue
#             box = track.to_tlwh()
#             x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
#             track_id = track.track_id
#             outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
#         if len(outputs) > 0:
#             outputs = np.stack(outputs,axis=0)
#         return outputs


#     """
#     TODO:
#         Convert bbox from xc_yc_w_h to xtl_ytl_w_h
#     Thanks JieChen91@github.com for reporting this bug!
#     """
#     @staticmethod
#     def _xywh_to_tlwh(bbox_xywh):
#         if isinstance(bbox_xywh, np.ndarray):
#             bbox_tlwh = bbox_xywh.copy()
#         elif isinstance(bbox_xywh, torch.Tensor):
#             bbox_tlwh = bbox_xywh.clone()
#         bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
#         bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
#         return bbox_tlwh


#     def _xywh_to_xyxy(self, bbox_xywh):
#         x,y,w,h = bbox_xywh
#         x1 = max(int(x-w/2),0)
#         x2 = min(int(x+w/2),self.width-1)
#         y1 = max(int(y-h/2),0)
#         y2 = min(int(y+h/2),self.height-1)
#         return x1,y1,x2,y2

#     def _tlwh_to_xyxy(self, bbox_tlwh):
#         """
#         TODO:
#             Convert bbox from xtl_ytl_w_h to xc_yc_w_h
#         Thanks JieChen91@github.com for reporting this bug!
#         """
#         x,y,w,h = bbox_tlwh
#         x1 = max(int(x),0)
#         x2 = min(int(x+w),self.width-1)
#         y1 = max(int(y),0)
#         y2 = min(int(y+h),self.height-1)
#         return x1,y1,x2,y2

#     def _xyxy_to_tlwh(self, bbox_xyxy):
#         x1,y1,x2,y2 = bbox_xyxy

#         t = x1
#         l = y1
#         w = int(x2-x1)
#         h = int(y2-y1)
#         return t,l,w,h
    
#     def _get_features(self, bbox_xywh, ori_img):
#         im_crops = []
#         for box in bbox_xywh:
#             x1,y1,x2,y2 = self._xywh_to_xyxy(box)
#             im = ori_img[y1:y2,x1:x2]
#             im_crops.append(im)
#         if im_crops:
#             features = self.extractor(im_crops)
#         else:
#             features = np.array([])
#         return features
    

#     def _maintain_track_status(self, track_bboxes, idx_frame):
#         for idx, each_active_tracks in enumerate(track_bboxes):
#             old_idx_frame = each_active_tracks[-1]
#             if idx_frame - old_idx_frame > self.config.threshold_dying_track_len:
#                 self.mark_old_track_idx.append(idx)

#     def cosine_similarity(self, a, b):
#         dot_product = np.dot(a, b)
#         norm_a = np.linalg.norm(a)
#         norm_b = np.linalg.norm(b)
#         similarity = dot_product / (norm_a * norm_b)
#         return similarity

#     def _compute_match_score(self, track_id, es_feature):
#         last_feature = self.all_es_feat_tracks[track_id][-1]
#         last_feature = np.array(last_feature)  # Convert to numpy array
        
#         try:
#             sim = self.cosine_similarity(last_feature.reshape(1, -1), np.array(es_feature).reshape(1, -1).T)
#         except:
#             pdb.set_trace()
#         if sim >= self.config.similarity_threshold:
#             return sim
#         else:
#             return -1
    
#     def _match_and_update_track(self, bbox, es_feature):
#         similar_track_ids = []
#         for idx, each_active_tracks in enumerate(self.all_tracks):
#             if idx in self.mark_old_track_idx:
#                 continue

#             # Compute the intersection over union (IoU) between the new bbox and the latest bbox of each existing track
#             iou = self.bbox_iou(bbox, each_active_tracks['bbox'][-1])

#             # If the IoU is above a threshold, add the existing track to the list of similar tracks
#             if iou > self.config.iou_threshold:
#                 similar_track_ids.append(idx)

#         if len(similar_track_ids) > 0:
#             # If there are similar tracks, update the one with the highest score
#             best_match_score = -1
#             for track_id in similar_track_ids:
#                 score = self._compute_match_score(track_id, es_feature)
#                 if score > best_match_score:
#                     best_match_score = score
#                     best_match_track_id = track_id

#         return best_match_track_id, best_match_score
    

#     def _check_exist_track(self, track_id):
#         where_track_id = None
#         for idx, _ in enumerate(self.all_tracks):
#             each_active_tracks = self.all_tracks[idx]
#             if each_active_tracks['id'] == track_id:
#                 where_track_id = idx
#                 break

#         return where_track_id

    
#     def _create_or_update(self, track_bboxes, emot_extractor, idx_frame, frame):
#         for track in track_bboxes:
#             x1, y1, x2, y2, track_id = track
            
#             bbox_tracker = [x1,y1,x2,y2]
            
#             # extract emotion feature
#             _, es_feature, emotion_cat = emot_extractor.run(frame, bbox_tracker)

#             exist_track_id = self._check_exist_track(track_id)
#             if exist_track_id is None:
#                 # create new track
#                 self._create_new_track(es_feature, emotion_cat, 
#                                      bbox_tracker, track_id, idx_frame)
                
#             else:
#                 # update new track
#                 self._update_old_track(es_feature, emotion_cat,
#                                        bbox_tracker, exist_track_id, idx_frame)

    
#     def _filter_tracks(self):
#         # filter those tracks having length smaller than a number
#         all_es_feat_tracks_filter = []
#         all_start_end_offset_track_filter = []
#         all_emotion_category_tracks_filter = []

#         for es_feat_track, se_track, ec_track in zip(self.all_es_feat_tracks, 
#                                                      self.all_start_end_offset_track, 
#                                                      self.all_emotion_category_tracks):
#             length = es_feat_track.shape[-1]
#             if length >= self.config.len_face_tracks:
#                 all_es_feat_tracks_filter.append(es_feat_track)
#                 all_start_end_offset_track_filter.append(se_track)
#                 all_emotion_category_tracks_filter.append(ec_track)

#         return all_es_feat_tracks_filter, all_start_end_offset_track_filter, all_emotion_category_tracks_filter

#     def _init_list_record(self):
#         self.all_tracks = []
#         self.mark_old_track_idx = []

#         self.all_emotion_category_tracks = []
#         self.all_es_feat_tracks = []
#         self.all_start_end_offset_track = []

#     def run(self, video, face_detector, emot_extractor, f_p_in):
#         self._init_list_record()
#         frames_list = list(video.iter_frames())
#         print(f_p_in)

#         for idx_frame, frame in tqdm(enumerate(frames_list), total=len(frames_list)):
#             if idx_frame % self.config.skip_frame != 0:
#                 continue      
            
#             # detect faces
#             bboxes, probs = face_detector.run(frame)
                        
#             if bboxes is None:
#                 continue
            
#             # Track the objects using DeepSORT
#             ## first convert xyxy to xc|yc|wh
#             # TO DO: Put this in a separate function
#             print('pre:', bboxes)
#             for idx, box in enumerate(bboxes):
#                 [x1, y1, x2, y2] = box

#                 width = x2 - x1
#                 height = y2 - y1

#                 xc = (x1 + width/2)
#                 yc = (y1 + height/2)

#                 bboxes[idx] = [xc, yc, width, height]

#             track_bboxes = self.update(bboxes, probs, frame)
#             print('af:', track_bboxes)

#             # pre: [[210.43867 146.40266 260.769   200.92578]]

#             # # Stage 1: Maintaining track status to kill inactive track
#             # self._maintain_track_status(track_bboxes, idx_frame)

#             # Stage 2: Assign new boxes to currently active tracks or create a new track if there are no active tracks
#             self._create_or_update(track_bboxes, emot_extractor, idx_frame, frame)

#             # debug mode
#             if self.config.is_debug:
#                 if idx_frame > self.config.max_frame_debug:
#                     self._debug_visualize_track_emotion(frames_list, f_p_in, 
#                                           video.fps, video.w, video.h)
#                     break
        
#         pdb.set_trace()
#         # get final result
#         all_es, all_se_offset, all_emot_cat = self._filter_tracks()

#         return all_es, all_se_offset, all_emot_cat




# # dont care the below class


# # iou matching
# def iou(bbox, candidates):
#     """Computer intersection over union.

#     Parameters
#     ----------
#     bbox : ndarray
#         A bounding box in format `(top left x, top left y, width, height)`.
#     candidates : ndarray
#         A matrix of candidate bounding boxes (one per row) in the same format
#         as `bbox`.

#     Returns
#     -------
#     ndarray
#         The intersection over union in [0, 1] between the `bbox` and each
#         candidate. A higher score means a larger fraction of the `bbox` is
#         occluded by the candidate.

#     """
#     bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
#     candidates_tl = candidates[:, :2]
#     candidates_br = candidates[:, :2] + candidates[:, 2:]

#     tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
#                np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
#     br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
#                np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
#     wh = np.maximum(0., br - tl)

#     area_intersection = wh.prod(axis=1)
#     area_bbox = bbox[2:].prod()
#     area_candidates = candidates[:, 2:].prod(axis=1)
#     return area_intersection / (area_bbox + area_candidates - area_intersection)


# def iou_cost(tracks, detections, track_indices=None,
#              detection_indices=None):
#     """An intersection over union distance metric.

#     Parameters
#     ----------
#     tracks : List[deep_sort.track.Track]
#         A list of tracks.
#     detections : List[deep_sort.detection.Detection]
#         A list of detections.
#     track_indices : Optional[List[int]]
#         A list of indices to tracks that should be matched. Defaults to
#         all `tracks`.
#     detection_indices : Optional[List[int]]
#         A list of indices to detections that should be matched. Defaults
#         to all `detections`.

#     Returns
#     -------
#     ndarray
#         Returns a cost matrix of shape
#         len(track_indices), len(detection_indices) where entry (i, j) is
#         `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

#     """
#     if track_indices is None:
#         track_indices = np.arange(len(tracks))
#     if detection_indices is None:
#         detection_indices = np.arange(len(detections))

#     cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
#     for row, track_idx in enumerate(track_indices):
#         if tracks[track_idx].time_since_update > 1:
#             cost_matrix[row, :] = INFTY_COST
#             continue

#         bbox = tracks[track_idx].to_tlwh()
#         candidates = np.asarray([detections[i].tlwh for i in detection_indices])
#         cost_matrix[row, :] = 1. - iou(bbox, candidates)
#     return cost_matrix

    
# # feature extractor
# class Extractor(object):
#     def __init__(self, model_path, use_cuda=True):
#         self.net = Net(reid=True)
#         self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
#         state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
#         self.net.load_state_dict(state_dict)
#         logger = logging.getLogger("root.tracker")
#         logger.info("Loading weights from {}... Done!".format(model_path))
#         self.net.to(self.device)
#         self.size = (64, 128)
#         self.norm = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         ])
        


#     def _preprocess(self, im_crops):
#         """
#         TODO:
#             1. to float with scale from 0 to 1
#             2. resize to (64, 128) as Market1501 dataset did
#             3. concatenate to a numpy array
#             3. to torch Tensor
#             4. normalize
#         """
#         def _resize(im, size):
#             return cv2.resize(im.astype(np.float32)/255., size)

#         im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
#         return im_batch


#     def __call__(self, im_crops):
#         im_batch = self._preprocess(im_crops)
#         with torch.no_grad():
#             im_batch = im_batch.to(self.device)
#             features = self.net(im_batch)
#         return features.cpu().numpy()

# class FastReIDExtractor(object):
#     def __init__(self, model_config, model_path, use_cuda=True):
#         cfg = get_cfg()
#         cfg.merge_from_file(model_config)
#         cfg.MODEL.BACKBONE.PRETRAIN = False
#         self.net = DefaultTrainer.build_model(cfg)
#         self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"

#         Checkpointer(self.net).load(model_path)
#         logger = logging.getLogger("root.tracker")
#         logger.info("Loading weights from {}... Done!".format(model_path))
#         self.net.to(self.device)
#         self.net.eval()
#         height, width = cfg.INPUT.SIZE_TEST
#         self.size = (width, height)
#         self.norm = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         ])

    
#     def _preprocess(self, im_crops):
#         def _resize(im, size):
#             return cv2.resize(im.astype(np.float32)/255., size)

#         im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
#         return im_batch


#     def __call__(self, im_crops):
#         im_batch = self._preprocess(im_crops)
#         with torch.no_grad():
#             im_batch = im_batch.to(self.device)
#             features = self.net(im_batch)
#         return features.cpu().numpy()
    

# # Network for deep sort
# class BasicBlock(nn.Module):
#     def __init__(self, c_in, c_out,is_downsample=False):
#         super(BasicBlock,self).__init__()
#         self.is_downsample = is_downsample
#         if is_downsample:
#             self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
#         else:
#             self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(c_out)
#         self.relu = nn.ReLU(True)
#         self.conv2 = nn.Conv2d(c_out,c_out,3,stride=1,padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(c_out)
#         if is_downsample:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
#                 nn.BatchNorm2d(c_out)
#             )
#         elif c_in != c_out:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
#                 nn.BatchNorm2d(c_out)
#             )
#             self.is_downsample = True

#     def forward(self,x):
#         y = self.conv1(x)
#         y = self.bn1(y)
#         y = self.relu(y)
#         y = self.conv2(y)
#         y = self.bn2(y)
#         if self.is_downsample:
#             x = self.downsample(x)
#         return F.relu(x.add(y),True)

# def make_layers(c_in,c_out,repeat_times, is_downsample=False):
#     blocks = []
#     for i in range(repeat_times):
#         if i ==0:
#             blocks += [BasicBlock(c_in,c_out, is_downsample=is_downsample),]
#         else:
#             blocks += [BasicBlock(c_out,c_out),]
#     return nn.Sequential(*blocks)

# class Net(nn.Module):
#     def __init__(self, num_classes=751 ,reid=False):
#         super(Net,self).__init__()
#         # 3 128 64
#         self.conv = nn.Sequential(
#             nn.Conv2d(3,64,3,stride=1,padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             # nn.Conv2d(32,32,3,stride=1,padding=1),
#             # nn.BatchNorm2d(32),
#             # nn.ReLU(inplace=True),
#             nn.MaxPool2d(3,2,padding=1),
#         )
#         # 32 64 32
#         self.layer1 = make_layers(64,64,2,False)
#         # 32 64 32
#         self.layer2 = make_layers(64,128,2,True)
#         # 64 32 16
#         self.layer3 = make_layers(128,256,2,True)
#         # 128 16 8
#         self.layer4 = make_layers(256,512,2,True)
#         # 256 8 4
#         self.avgpool = nn.AvgPool2d((8,4),1)
#         # 256 1 1 
#         self.reid = reid
#         self.classifier = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(256, num_classes),
#         )
    
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0),-1)
#         # B x 128
#         if self.reid:
#             x = x.div(x.norm(p=2,dim=1,keepdim=True))
#             return x
#         # classifier
#         x = self.classifier(x)
#         return x
    

# # nn matching
# def _pdist(a, b):
#     """Compute pair-wise squared distance between points in `a` and `b`.

#     Parameters
#     ----------
#     a : array_like
#         An NxM matrix of N samples of dimensionality M.
#     b : array_like
#         An LxM matrix of L samples of dimensionality M.

#     Returns
#     -------
#     ndarray
#         Returns a matrix of size len(a), len(b) such that eleement (i, j)
#         contains the squared distance between `a[i]` and `b[j]`.

#     """
#     a, b = np.asarray(a), np.asarray(b)
#     if len(a) == 0 or len(b) == 0:
#         return np.zeros((len(a), len(b)))
#     a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
#     r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
#     r2 = np.clip(r2, 0., float(np.inf))
#     return r2


# def _cosine_distance(a, b, data_is_normalized=False):
#     """Compute pair-wise cosine distance between points in `a` and `b`.

#     Parameters
#     ----------
#     a : array_like
#         An NxM matrix of N samples of dimensionality M.
#     b : array_like
#         An LxM matrix of L samples of dimensionality M.
#     data_is_normalized : Optional[bool]
#         If True, assumes rows in a and b are unit length vectors.
#         Otherwise, a and b are explicitly normalized to lenght 1.

#     Returns
#     -------
#     ndarray
#         Returns a matrix of size len(a), len(b) such that eleement (i, j)
#         contains the squared distance between `a[i]` and `b[j]`.

#     """
#     if not data_is_normalized:
#         a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
#         b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
#     return 1. - np.dot(a, b.T)


# def _nn_euclidean_distance(x, y):
#     """ Helper function for nearest neighbor distance metric (Euclidean).

#     Parameters
#     ----------
#     x : ndarray
#         A matrix of N row-vectors (sample points).
#     y : ndarray
#         A matrix of M row-vectors (query points).

#     Returns
#     -------
#     ndarray
#         A vector of length M that contains for each entry in `y` the
#         smallest Euclidean distance to a sample in `x`.

#     """
#     distances = _pdist(x, y)
#     return np.maximum(0.0, distances.min(axis=0))


# def _nn_cosine_distance(x, y):
#     """ Helper function for nearest neighbor distance metric (cosine).

#     Parameters
#     ----------
#     x : ndarray
#         A matrix of N row-vectors (sample points).
#     y : ndarray
#         A matrix of M row-vectors (query points).

#     Returns
#     -------
#     ndarray
#         A vector of length M that contains for each entry in `y` the
#         smallest cosine distance to a sample in `x`.

#     """
#     distances = _cosine_distance(x, y)
#     return distances.min(axis=0)


# class NearestNeighborDistanceMetric(object):
#     """
#     A nearest neighbor distance metric that, for each target, returns
#     the closest distance to any sample that has been observed so far.

#     Parameters
#     ----------
#     metric : str
#         Either "euclidean" or "cosine".
#     matching_threshold: float
#         The matching threshold. Samples with larger distance are considered an
#         invalid match.
#     budget : Optional[int]
#         If not None, fix samples per class to at most this number. Removes
#         the oldest samples when the budget is reached.

#     Attributes
#     ----------
#     samples : Dict[int -> List[ndarray]]
#         A dictionary that maps from target identities to the list of samples
#         that have been observed so far.

#     """

#     def __init__(self, metric, matching_threshold, budget=None):


#         if metric == "euclidean":
#             self._metric = _nn_euclidean_distance
#         elif metric == "cosine":
#             self._metric = _nn_cosine_distance
#         else:
#             raise ValueError(
#                 "Invalid metric; must be either 'euclidean' or 'cosine'")
#         self.matching_threshold = matching_threshold
#         self.budget = budget
#         self.samples = {}

#     def partial_fit(self, features, targets, active_targets):
#         """Update the distance metric with new data.

#         Parameters
#         ----------
#         features : ndarray
#             An NxM matrix of N features of dimensionality M.
#         targets : ndarray
#             An integer array of associated target identities.
#         active_targets : List[int]
#             A list of targets that are currently present in the scene.

#         """
#         for feature, target in zip(features, targets):
#             self.samples.setdefault(target, []).append(feature)
#             if self.budget is not None:
#                 self.samples[target] = self.samples[target][-self.budget:]
#         self.samples = {k: self.samples[k] for k in active_targets}

#     def distance(self, features, targets):
#         """Compute distance between features and targets.

#         Parameters
#         ----------
#         features : ndarray
#             An NxM matrix of N features of dimensionality M.
#         targets : List[int]
#             A list of targets to match the given `features` against.

#         Returns
#         -------
#         ndarray
#             Returns a cost matrix of shape len(targets), len(features), where
#             element (i, j) contains the closest squared distance between
#             `targets[i]` and `features[j]`.

#         """
#         cost_matrix = np.zeros((len(targets), len(features)))
#         for i, target in enumerate(targets):
#             cost_matrix[i, :] = self._metric(self.samples[target], features)
#         return cost_matrix


# # preprocessing
# def non_max_suppression(boxes, max_bbox_overlap, scores=None):
#     """Suppress overlapping detections.

#     Original code from [1]_ has been adapted to include confidence score.

#     .. [1] http://www.pyimagesearch.com/2015/02/16/
#            faster-non-maximum-suppression-python/

#     Examples
#     --------

#         >>> boxes = [d.roi for d in detections]
#         >>> scores = [d.confidence for d in detections]
#         >>> indices = non_max_suppression(boxes, max_bbox_overlap, scores)
#         >>> detections = [detections[i] for i in indices]

#     Parameters
#     ----------
#     boxes : ndarray
#         Array of ROIs (x, y, width, height).
#     max_bbox_overlap : float
#         ROIs that overlap more than this values are suppressed.
#     scores : Optional[array_like]
#         Detector confidence score.

#     Returns
#     -------
#     List[int]
#         Returns indices of detections that have survived non-maxima suppression.

#     """
#     if len(boxes) == 0:
#         return []

#     boxes = boxes.astype(np.float)
#     pick = []

#     x1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     x2 = boxes[:, 2] + boxes[:, 0]
#     y2 = boxes[:, 3] + boxes[:, 1]

#     area = (x2 - x1 + 1) * (y2 - y1 + 1)
#     if scores is not None:
#         idxs = np.argsort(scores)
#     else:
#         idxs = np.argsort(y2)

#     while len(idxs) > 0:
#         last = len(idxs) - 1
#         i = idxs[last]
#         pick.append(i)

#         xx1 = np.maximum(x1[i], x1[idxs[:last]])
#         yy1 = np.maximum(y1[i], y1[idxs[:last]])
#         xx2 = np.minimum(x2[i], x2[idxs[:last]])
#         yy2 = np.minimum(y2[i], y2[idxs[:last]])

#         w = np.maximum(0, xx2 - xx1 + 1)
#         h = np.maximum(0, yy2 - yy1 + 1)

#         overlap = (w * h) / area[idxs[:last]]

#         idxs = np.delete(
#             idxs, np.concatenate(
#                 ([last], np.where(overlap > max_bbox_overlap)[0])))

#     return pick


# # detection
# class Detection(object):
#     """
#     This class represents a bounding box detection in a single image.

#     Parameters
#     ----------
#     tlwh : array_like
#         Bounding box in format `(x, y, w, h)`.
#     confidence : float
#         Detector confidence score.
#     feature : array_like
#         A feature vector that describes the object contained in this image.

#     Attributes
#     ----------
#     tlwh : ndarray
#         Bounding box in format `(top left x, top left y, width, height)`.
#     confidence : ndarray
#         Detector confidence score.
#     feature : ndarray | NoneType
#         A feature vector that describes the object contained in this image.

#     """

#     def __init__(self, tlwh, confidence, feature):
#         self.tlwh = np.asarray(tlwh, dtype=np.float)
#         self.confidence = float(confidence)
#         self.feature = np.asarray(feature, dtype=np.float32)

#     def to_tlbr(self):
#         """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
#         `(top left, bottom right)`.
#         """
#         ret = self.tlwh.copy()
#         ret[2:] += ret[:2]
#         return ret

#     def to_xyah(self):
#         """Convert bounding box to format `(center x, center y, aspect ratio,
#         height)`, where the aspect ratio is `width / height`.
#         """
#         ret = self.tlwh.copy()
#         ret[:2] += ret[2:] / 2
#         ret[2] /= ret[3]
#         return ret
    

# # tracker
# class TrackState:
#     """
#     Enumeration type for the single target track state. Newly created tracks are
#     classified as `tentative` until enough evidence has been collected. Then,
#     the track state is changed to `confirmed`. Tracks that are no longer alive
#     are classified as `deleted` to mark them for removal from the set of active
#     tracks.

#     """

#     Tentative = 1
#     Confirmed = 2
#     Deleted = 3


# class Track:
#     """
#     A single target track with state space `(x, y, a, h)` and associated
#     velocities, where `(x, y)` is the center of the bounding box, `a` is the
#     aspect ratio and `h` is the height.

#     Parameters
#     ----------
#     mean : ndarray
#         Mean vector of the initial state distribution.
#     covariance : ndarray
#         Covariance matrix of the initial state distribution.
#     track_id : int
#         A unique track identifier.
#     n_init : int
#         Number of consecutive detections before the track is confirmed. The
#         track state is set to `Deleted` if a miss occurs within the first
#         `n_init` frames.
#     max_age : int
#         The maximum number of consecutive misses before the track state is
#         set to `Deleted`.
#     feature : Optional[ndarray]
#         Feature vector of the detection this track originates from. If not None,
#         this feature is added to the `features` cache.

#     Attributes
#     ----------
#     mean : ndarray
#         Mean vector of the initial state distribution.
#     covariance : ndarray
#         Covariance matrix of the initial state distribution.
#     track_id : int
#         A unique track identifier.
#     hits : int
#         Total number of measurement updates.
#     age : int
#         Total number of frames since first occurance.
#     time_since_update : int
#         Total number of frames since last measurement update.
#     state : TrackState
#         The current track state.
#     features : List[ndarray]
#         A cache of features. On each measurement update, the associated feature
#         vector is added to this list.

#     """

#     def __init__(self, mean, covariance, track_id, n_init, max_age,
#                  feature=None):
#         self.mean = mean
#         self.covariance = covariance
#         self.track_id = track_id
#         self.hits = 1
#         self.age = 1
#         self.time_since_update = 0

#         self.state = TrackState.Tentative
#         self.features = []
#         if feature is not None:
#             self.features.append(feature)

#         self._n_init = n_init
#         self._max_age = max_age

#     def to_tlwh(self):
#         """Get current position in bounding box format `(top left x, top left y,
#         width, height)`.

#         Returns
#         -------
#         ndarray
#             The bounding box.

#         """
#         ret = self.mean[:4].copy()
#         ret[2] *= ret[3]
#         ret[:2] -= ret[2:] / 2
#         return ret

#     def to_tlbr(self):
#         """Get current position in bounding box format `(min x, miny, max x,
#         max y)`.

#         Returns
#         -------
#         ndarray
#             The bounding box.

#         """
#         ret = self.to_tlwh()
#         ret[2:] = ret[:2] + ret[2:]
#         return ret
    
#     def to_x1y1x2y2(self):
#         """BAO get current position in bounding box format `(x1, y1, x2, y2)`.

#         Returns
#         -------
#         ndarray
#             The bounding box.

#         """
#         ret = self.mean[:4].copy()
#         ret[2] *= ret[3]

#         return ret

#     def predict(self, kf):
#         """Propagate the state distribution to the current time step using a
#         Kalman filter prediction step.

#         Parameters
#         ----------
#         kf : kalman_filter.KalmanFilter
#             The Kalman filter.

#         """
#         self.mean, self.covariance = kf.predict(self.mean, self.covariance)
#         self.age += 1
#         self.time_since_update += 1

#     def update(self, kf, detection):
#         """Perform Kalman filter measurement update step and update the feature
#         cache.

#         Parameters
#         ----------
#         kf : kalman_filter.KalmanFilter
#             The Kalman filter.
#         detection : Detection
#             The associated detection.

#         """
#         self.mean, self.covariance = kf.update(
#             self.mean, self.covariance, detection.to_xyah())
#         self.features.append(detection.feature)

#         self.hits += 1
#         self.time_since_update = 0
#         if self.state == TrackState.Tentative and self.hits >= self._n_init:
#             self.state = TrackState.Confirmed

#     def mark_missed(self):
#         """Mark this track as missed (no association at the current time step).
#         """
#         if self.state == TrackState.Tentative:
#             self.state = TrackState.Deleted
#         elif self.time_since_update > self._max_age:
#             self.state = TrackState.Deleted

#     def is_tentative(self):
#         """Returns True if this track is tentative (unconfirmed).
#         """
#         return self.state == TrackState.Tentative

#     def is_confirmed(self):
#         """Returns True if this track is confirmed."""
#         return self.state == TrackState.Confirmed

#     def is_deleted(self):
#         """Returns True if this track is dead and should be deleted."""
#         return self.state == TrackState.Deleted

# class Tracker:
#     """
#     This is the multi-target tracker.

#     Parameters
#     ----------
#     metric : nn_matching.NearestNeighborDistanceMetric
#         A distance metric for measurement-to-track association.
#     max_age : int
#         Maximum number of missed misses before a track is deleted.
#     n_init : int
#         Number of consecutive detections before the track is confirmed. The
#         track state is set to `Deleted` if a miss occurs within the first
#         `n_init` frames.

#     Attributes
#     ----------
#     metric : nn_matching.NearestNeighborDistanceMetric
#         The distance metric used for measurement to track association.
#     max_age : int
#         Maximum number of missed misses before a track is deleted.
#     n_init : int
#         Number of frames that a track remains in initialization phase.
#     kf : kalman_filter.KalmanFilter
#         A Kalman filter to filter target trajectories in image space.
#     tracks : List[Track]
#         The list of active tracks at the current time step.

#     """

#     def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
#         self.metric = metric
#         self.max_iou_distance = max_iou_distance
#         self.max_age = max_age
#         self.n_init = n_init

#         self.kf = KalmanFilter()
#         self.tracks = []
#         self._next_id = 1

#     def predict(self):
#         """Propagate track state distributions one time step forward.

#         This function should be called once every time step, before `update`.
#         """
#         for track in self.tracks:
#             track.predict(self.kf)

#     def update(self, detections):
#         """Perform measurement update and track management.

#         Parameters
#         ----------
#         detections : List[deep_sort.detection.Detection]
#             A list of detections at the current time step.

#         """
#         # Run matching cascade.
#         matches, unmatched_tracks, unmatched_detections = \
#             self._match(detections)

#         # Update track set.
#         for track_idx, detection_idx in matches:
#             self.tracks[track_idx].update(
#                 self.kf, detections[detection_idx])
#         for track_idx in unmatched_tracks:
#             self.tracks[track_idx].mark_missed()
#         for detection_idx in unmatched_detections:
#             self._initiate_track(detections[detection_idx])
#         self.tracks = [t for t in self.tracks if not t.is_deleted()]

#         # Update distance metric.
#         active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
#         features, targets = [], []
#         for track in self.tracks:
#             if not track.is_confirmed():
#                 continue
#             features += track.features
#             targets += [track.track_id for _ in track.features]
#             track.features = []
#         self.metric.partial_fit(
#             np.asarray(features), np.asarray(targets), active_targets)

#     def _match(self, detections):

#         def gated_metric(tracks, dets, track_indices, detection_indices):
#             features = np.array([dets[i].feature for i in detection_indices])
#             targets = np.array([tracks[i].track_id for i in track_indices])
#             cost_matrix = self.metric.distance(features, targets)
#             cost_matrix = gate_cost_matrix(
#                 self.kf, cost_matrix, tracks, dets, track_indices,
#                 detection_indices)

#             return cost_matrix

#         # Split track set into confirmed and unconfirmed tracks.
#         confirmed_tracks = [
#             i for i, t in enumerate(self.tracks) if t.is_confirmed()]
#         unconfirmed_tracks = [
#             i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

#         # Associate confirmed tracks using appearance features.
#         matches_a, unmatched_tracks_a, unmatched_detections = \
#                 matching_cascade(
#                 gated_metric, self.metric.matching_threshold, self.max_age,
#                 self.tracks, detections, confirmed_tracks)

#         # Associate remaining tracks together with unconfirmed tracks using IOU.
#         iou_track_candidates = unconfirmed_tracks + [
#             k for k in unmatched_tracks_a if
#             self.tracks[k].time_since_update == 1]
#         unmatched_tracks_a = [
#             k for k in unmatched_tracks_a if
#             self.tracks[k].time_since_update != 1]
#         matches_b, unmatched_tracks_b, unmatched_detections = \
#                 min_cost_matching(
#                 iou_cost, self.max_iou_distance, self.tracks,
#                 detections, iou_track_candidates, unmatched_detections)

#         matches = matches_a + matches_b
#         unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
#         return matches, unmatched_tracks, unmatched_detections

#     def _initiate_track(self, detection):
#         mean, covariance = self.kf.initiate(detection.to_xyah())
#         self.tracks.append(Track(
#             mean, covariance, self._next_id, self.n_init, self.max_age,
#             detection.feature))
#         self._next_id += 1

# # linear assignmetn
# def min_cost_matching(
#         distance_metric, max_distance, tracks, detections, track_indices=None,
#         detection_indices=None):
#     """Solve linear assignment problem.

#     Parameters
#     ----------
#     distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
#         The distance metric is given a list of tracks and detections as well as
#         a list of N track indices and M detection indices. The metric should
#         return the NxM dimensional cost matrix, where element (i, j) is the
#         association cost between the i-th track in the given track indices and
#         the j-th detection in the given detection_indices.
#     max_distance : float
#         Gating threshold. Associations with cost larger than this value are
#         disregarded.
#     tracks : List[track.Track]
#         A list of predicted tracks at the current time step.
#     detections : List[detection.Detection]
#         A list of detections at the current time step.
#     track_indices : List[int]
#         List of track indices that maps rows in `cost_matrix` to tracks in
#         `tracks` (see description above).
#     detection_indices : List[int]
#         List of detection indices that maps columns in `cost_matrix` to
#         detections in `detections` (see description above).

#     Returns
#     -------
#     (List[(int, int)], List[int], List[int])
#         Returns a tuple with the following three entries:
#         * A list of matched track and detection indices.
#         * A list of unmatched track indices.
#         * A list of unmatched detection indices.

#     """
#     if track_indices is None:
#         track_indices = np.arange(len(tracks))
#     if detection_indices is None:
#         detection_indices = np.arange(len(detections))

#     if len(detection_indices) == 0 or len(track_indices) == 0:
#         return [], track_indices, detection_indices  # Nothing to match.

#     cost_matrix = distance_metric(
#         tracks, detections, track_indices, detection_indices)
#     cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

#     row_indices, col_indices = linear_assignment(cost_matrix)

#     matches, unmatched_tracks, unmatched_detections = [], [], []
#     for col, detection_idx in enumerate(detection_indices):
#         if col not in col_indices:
#             unmatched_detections.append(detection_idx)
#     for row, track_idx in enumerate(track_indices):
#         if row not in row_indices:
#             unmatched_tracks.append(track_idx)
#     for row, col in zip(row_indices, col_indices):
#         track_idx = track_indices[row]
#         detection_idx = detection_indices[col]
#         if cost_matrix[row, col] > max_distance:
#             unmatched_tracks.append(track_idx)
#             unmatched_detections.append(detection_idx)
#         else:
#             matches.append((track_idx, detection_idx))
#     return matches, unmatched_tracks, unmatched_detections


# def matching_cascade(
#         distance_metric, max_distance, cascade_depth, tracks, detections,
#         track_indices=None, detection_indices=None):
#     """Run matching cascade.

#     Parameters
#     ----------
#     distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
#         The distance metric is given a list of tracks and detections as well as
#         a list of N track indices and M detection indices. The metric should
#         return the NxM dimensional cost matrix, where element (i, j) is the
#         association cost between the i-th track in the given track indices and
#         the j-th detection in the given detection indices.
#     max_distance : float
#         Gating threshold. Associations with cost larger than this value are
#         disregarded.
#     cascade_depth: int
#         The cascade depth, should be se to the maximum track age.
#     tracks : List[track.Track]
#         A list of predicted tracks at the current time step.
#     detections : List[detection.Detection]
#         A list of detections at the current time step.
#     track_indices : Optional[List[int]]
#         List of track indices that maps rows in `cost_matrix` to tracks in
#         `tracks` (see description above). Defaults to all tracks.
#     detection_indices : Optional[List[int]]
#         List of detection indices that maps columns in `cost_matrix` to
#         detections in `detections` (see description above). Defaults to all
#         detections.

#     Returns
#     -------
#     (List[(int, int)], List[int], List[int])
#         Returns a tuple with the following three entries:
#         * A list of matched track and detection indices.
#         * A list of unmatched track indices.
#         * A list of unmatched detection indices.

#     """
#     if track_indices is None:
#         track_indices = list(range(len(tracks)))
#     if detection_indices is None:
#         detection_indices = list(range(len(detections)))

#     unmatched_detections = detection_indices
#     matches = []
#     for level in range(cascade_depth):
#         if len(unmatched_detections) == 0:  # No detections left
#             break

#         track_indices_l = [
#             k for k in track_indices
#             if tracks[k].time_since_update == 1 + level
#         ]
#         if len(track_indices_l) == 0:  # Nothing to match at this level
#             continue

#         matches_l, _, unmatched_detections = \
#             min_cost_matching(
#                 distance_metric, max_distance, tracks, detections,
#                 track_indices_l, unmatched_detections)
#         matches += matches_l
#     unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
#     return matches, unmatched_tracks, unmatched_detections


# def gate_cost_matrix(
#         kf, cost_matrix, tracks, detections, track_indices, detection_indices,
#         gated_cost=INFTY_COST, only_position=False):
#     """Invalidate infeasible entries in cost matrix based on the state
#     distributions obtained by Kalman filtering.

#     Parameters
#     ----------
#     kf : The Kalman filter.
#     cost_matrix : ndarray
#         The NxM dimensional cost matrix, where N is the number of track indices
#         and M is the number of detection indices, such that entry (i, j) is the
#         association cost between `tracks[track_indices[i]]` and
#         `detections[detection_indices[j]]`.
#     tracks : List[track.Track]
#         A list of predicted tracks at the current time step.
#     detections : List[detection.Detection]
#         A list of detections at the current time step.
#     track_indices : List[int]
#         List of track indices that maps rows in `cost_matrix` to tracks in
#         `tracks` (see description above).
#     detection_indices : List[int]
#         List of detection indices that maps columns in `cost_matrix` to
#         detections in `detections` (see description above).
#     gated_cost : Optional[float]
#         Entries in the cost matrix corresponding to infeasible associations are
#         set this value. Defaults to a very large value.
#     only_position : Optional[bool]
#         If True, only the x, y position of the state distribution is considered
#         during gating. Defaults to False.

#     Returns
#     -------
#     ndarray
#         Returns the modified cost matrix.

#     """
#     gating_dim = 2 if only_position else 4
#     gating_threshold = chi2inv95[gating_dim]
#     measurements = np.asarray(
#         [detections[i].to_xyah() for i in detection_indices])
#     for row, track_idx in enumerate(track_indices):
#         track = tracks[track_idx]
#         gating_distance = kf.gating_distance(
#             track.mean, track.covariance, measurements, only_position)
#         cost_matrix[row, gating_distance > gating_threshold] = gated_cost
#     return cost_matrix

# INFTY_COST = 1e+5
# # linear assignment
# def min_cost_matching(
#         distance_metric, max_distance, tracks, detections, track_indices=None,
#         detection_indices=None):
#     """Solve linear assignment problem.

#     Parameters
#     ----------
#     distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
#         The distance metric is given a list of tracks and detections as well as
#         a list of N track indices and M detection indices. The metric should
#         return the NxM dimensional cost matrix, where element (i, j) is the
#         association cost between the i-th track in the given track indices and
#         the j-th detection in the given detection_indices.
#     max_distance : float
#         Gating threshold. Associations with cost larger than this value are
#         disregarded.
#     tracks : List[track.Track]
#         A list of predicted tracks at the current time step.
#     detections : List[detection.Detection]
#         A list of detections at the current time step.
#     track_indices : List[int]
#         List of track indices that maps rows in `cost_matrix` to tracks in
#         `tracks` (see description above).
#     detection_indices : List[int]
#         List of detection indices that maps columns in `cost_matrix` to
#         detections in `detections` (see description above).

#     Returns
#     -------
#     (List[(int, int)], List[int], List[int])
#         Returns a tuple with the following three entries:
#         * A list of matched track and detection indices.
#         * A list of unmatched track indices.
#         * A list of unmatched detection indices.

#     """
#     if track_indices is None:
#         track_indices = np.arange(len(tracks))
#     if detection_indices is None:
#         detection_indices = np.arange(len(detections))

#     if len(detection_indices) == 0 or len(track_indices) == 0:
#         return [], track_indices, detection_indices  # Nothing to match.

#     cost_matrix = distance_metric(
#         tracks, detections, track_indices, detection_indices)
#     cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

#     row_indices, col_indices = linear_assignment(cost_matrix)

#     matches, unmatched_tracks, unmatched_detections = [], [], []
#     for col, detection_idx in enumerate(detection_indices):
#         if col not in col_indices:
#             unmatched_detections.append(detection_idx)
#     for row, track_idx in enumerate(track_indices):
#         if row not in row_indices:
#             unmatched_tracks.append(track_idx)
#     for row, col in zip(row_indices, col_indices):
#         track_idx = track_indices[row]
#         detection_idx = detection_indices[col]
#         if cost_matrix[row, col] > max_distance:
#             unmatched_tracks.append(track_idx)
#             unmatched_detections.append(detection_idx)
#         else:
#             matches.append((track_idx, detection_idx))
#     return matches, unmatched_tracks, unmatched_detections


# def matching_cascade(
#         distance_metric, max_distance, cascade_depth, tracks, detections,
#         track_indices=None, detection_indices=None):
#     """Run matching cascade.

#     Parameters
#     ----------
#     distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
#         The distance metric is given a list of tracks and detections as well as
#         a list of N track indices and M detection indices. The metric should
#         return the NxM dimensional cost matrix, where element (i, j) is the
#         association cost between the i-th track in the given track indices and
#         the j-th detection in the given detection indices.
#     max_distance : float
#         Gating threshold. Associations with cost larger than this value are
#         disregarded.
#     cascade_depth: int
#         The cascade depth, should be se to the maximum track age.
#     tracks : List[track.Track]
#         A list of predicted tracks at the current time step.
#     detections : List[detection.Detection]
#         A list of detections at the current time step.
#     track_indices : Optional[List[int]]
#         List of track indices that maps rows in `cost_matrix` to tracks in
#         `tracks` (see description above). Defaults to all tracks.
#     detection_indices : Optional[List[int]]
#         List of detection indices that maps columns in `cost_matrix` to
#         detections in `detections` (see description above). Defaults to all
#         detections.

#     Returns
#     -------
#     (List[(int, int)], List[int], List[int])
#         Returns a tuple with the following three entries:
#         * A list of matched track and detection indices.
#         * A list of unmatched track indices.
#         * A list of unmatched detection indices.

#     """
#     if track_indices is None:
#         track_indices = list(range(len(tracks)))
#     if detection_indices is None:
#         detection_indices = list(range(len(detections)))

#     unmatched_detections = detection_indices
#     matches = []
#     for level in range(cascade_depth):
#         if len(unmatched_detections) == 0:  # No detections left
#             break

#         track_indices_l = [
#             k for k in track_indices
#             if tracks[k].time_since_update == 1 + level
#         ]
#         if len(track_indices_l) == 0:  # Nothing to match at this level
#             continue

#         matches_l, _, unmatched_detections = \
#             min_cost_matching(
#                 distance_metric, max_distance, tracks, detections,
#                 track_indices_l, unmatched_detections)
#         matches += matches_l
#     unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
#     return matches, unmatched_tracks, unmatched_detections


# def gate_cost_matrix(
#         kf, cost_matrix, tracks, detections, track_indices, detection_indices,
#         gated_cost=INFTY_COST, only_position=False):
#     """Invalidate infeasible entries in cost matrix based on the state
#     distributions obtained by Kalman filtering.

#     Parameters
#     ----------
#     kf : The Kalman filter.
#     cost_matrix : ndarray
#         The NxM dimensional cost matrix, where N is the number of track indices
#         and M is the number of detection indices, such that entry (i, j) is the
#         association cost between `tracks[track_indices[i]]` and
#         `detections[detection_indices[j]]`.
#     tracks : List[track.Track]
#         A list of predicted tracks at the current time step.
#     detections : List[detection.Detection]
#         A list of detections at the current time step.
#     track_indices : List[int]
#         List of track indices that maps rows in `cost_matrix` to tracks in
#         `tracks` (see description above).
#     detection_indices : List[int]
#         List of detection indices that maps columns in `cost_matrix` to
#         detections in `detections` (see description above).
#     gated_cost : Optional[float]
#         Entries in the cost matrix corresponding to infeasible associations are
#         set this value. Defaults to a very large value.
#     only_position : Optional[bool]
#         If True, only the x, y position of the state distribution is considered
#         during gating. Defaults to False.

#     Returns
#     -------
#     ndarray
#         Returns the modified cost matrix.

#     """
#     gating_dim = 2 if only_position else 4
#     gating_threshold = chi2inv95[gating_dim]
#     measurements = np.asarray(
#         [detections[i].to_xyah() for i in detection_indices])
#     for row, track_idx in enumerate(track_indices):
#         track = tracks[track_idx]
#         gating_distance = kf.gating_distance(
#             track.mean, track.covariance, measurements, only_position)
#         cost_matrix[row, gating_distance > gating_threshold] = gated_cost
#     return cost_matrix

# # kalman filter
# """
# Table for the 0.95 quantile of the chi-square distribution with N degrees of
# freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
# function and used as Mahalanobis gating threshold.
# """
# chi2inv95 = {
#     1: 3.8415,
#     2: 5.9915,
#     3: 7.8147,
#     4: 9.4877,
#     5: 11.070,
#     6: 12.592,
#     7: 14.067,
#     8: 15.507,
#     9: 16.919}


# class KalmanFilter(object):
#     """
#     A simple Kalman filter for tracking bounding boxes in image space.

#     The 8-dimensional state space

#         x, y, a, h, vx, vy, va, vh

#     contains the bounding box center position (x, y), aspect ratio a, height h,
#     and their respective velocities.

#     Object motion follows a constant velocity model. The bounding box location
#     (x, y, a, h) is taken as direct observation of the state space (linear
#     observation model).

#     """

#     def __init__(self):
#         ndim, dt = 4, 1.

#         # Create Kalman filter model matrices.
#         self._motion_mat = np.eye(2 * ndim, 2 * ndim)
#         for i in range(ndim):
#             self._motion_mat[i, ndim + i] = dt
#         self._update_mat = np.eye(ndim, 2 * ndim)

#         # Motion and observation uncertainty are chosen relative to the current
#         # state estimate. These weights control the amount of uncertainty in
#         # the model. This is a bit hacky.
#         self._std_weight_position = 1. / 20
#         self._std_weight_velocity = 1. / 160

#     def initiate(self, measurement):
#         """Create track from unassociated measurement.

#         Parameters
#         ----------
#         measurement : ndarray
#             Bounding box coordinates (x, y, a, h) with center position (x, y),
#             aspect ratio a, and height h.

#         Returns
#         -------
#         (ndarray, ndarray)
#             Returns the mean vector (8 dimensional) and covariance matrix (8x8
#             dimensional) of the new track. Unobserved velocities are initialized
#             to 0 mean.

#         """
#         mean_pos = measurement
#         mean_vel = np.zeros_like(mean_pos)
#         mean = np.r_[mean_pos, mean_vel]

#         std = [
#             2 * self._std_weight_position * measurement[3],
#             2 * self._std_weight_position * measurement[3],
#             1e-2,
#             2 * self._std_weight_position * measurement[3],
#             10 * self._std_weight_velocity * measurement[3],
#             10 * self._std_weight_velocity * measurement[3],
#             1e-5,
#             10 * self._std_weight_velocity * measurement[3]]
#         covariance = np.diag(np.square(std))
#         return mean, covariance

#     def predict(self, mean, covariance):
#         """Run Kalman filter prediction step.

#         Parameters
#         ----------
#         mean : ndarray
#             The 8 dimensional mean vector of the object state at the previous
#             time step.
#         covariance : ndarray
#             The 8x8 dimensional covariance matrix of the object state at the
#             previous time step.

#         Returns
#         -------
#         (ndarray, ndarray)
#             Returns the mean vector and covariance matrix of the predicted
#             state. Unobserved velocities are initialized to 0 mean.

#         """
#         std_pos = [
#             self._std_weight_position * mean[3],
#             self._std_weight_position * mean[3],
#             1e-2,
#             self._std_weight_position * mean[3]]
#         std_vel = [
#             self._std_weight_velocity * mean[3],
#             self._std_weight_velocity * mean[3],
#             1e-5,
#             self._std_weight_velocity * mean[3]]
#         motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

#         mean = np.dot(self._motion_mat, mean)
#         covariance = np.linalg.multi_dot((
#             self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

#         return mean, covariance

#     def project(self, mean, covariance):
#         """Project state distribution to measurement space.

#         Parameters
#         ----------
#         mean : ndarray
#             The state's mean vector (8 dimensional array).
#         covariance : ndarray
#             The state's covariance matrix (8x8 dimensional).

#         Returns
#         -------
#         (ndarray, ndarray)
#             Returns the projected mean and covariance matrix of the given state
#             estimate.

#         """
#         std = [
#             self._std_weight_position * mean[3],
#             self._std_weight_position * mean[3],
#             1e-1,
#             self._std_weight_position * mean[3]]
#         innovation_cov = np.diag(np.square(std))

#         mean = np.dot(self._update_mat, mean)
#         covariance = np.linalg.multi_dot((
#             self._update_mat, covariance, self._update_mat.T))
#         return mean, covariance + innovation_cov

#     def update(self, mean, covariance, measurement):
#         """Run Kalman filter correction step.

#         Parameters
#         ----------
#         mean : ndarray
#             The predicted state's mean vector (8 dimensional).
#         covariance : ndarray
#             The state's covariance matrix (8x8 dimensional).
#         measurement : ndarray
#             The 4 dimensional measurement vector (x, y, a, h), where (x, y)
#             is the center position, a the aspect ratio, and h the height of the
#             bounding box.

#         Returns
#         -------
#         (ndarray, ndarray)
#             Returns the measurement-corrected state distribution.

#         """
#         projected_mean, projected_cov = self.project(mean, covariance)

#         chol_factor, lower = scipy.linalg.cho_factor(
#             projected_cov, lower=True, check_finite=False)
#         kalman_gain = scipy.linalg.cho_solve(
#             (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
#             check_finite=False).T
#         innovation = measurement - projected_mean

#         new_mean = mean + np.dot(innovation, kalman_gain.T)
#         new_covariance = covariance - np.linalg.multi_dot((
#             kalman_gain, projected_cov, kalman_gain.T))
#         return new_mean, new_covariance

#     def gating_distance(self, mean, covariance, measurements,
#                         only_position=False):
#         """Compute gating distance between state distribution and measurements.

#         A suitable distance threshold can be obtained from `chi2inv95`. If
#         `only_position` is False, the chi-square distribution has 4 degrees of
#         freedom, otherwise 2.

#         Parameters
#         ----------
#         mean : ndarray
#             Mean vector over the state distribution (8 dimensional).
#         covariance : ndarray
#             Covariance of the state distribution (8x8 dimensional).
#         measurements : ndarray
#             An Nx4 dimensional matrix of N measurements, each in
#             format (x, y, a, h) where (x, y) is the bounding box center
#             position, a the aspect ratio, and h the height.
#         only_position : Optional[bool]
#             If True, distance computation is done with respect to the bounding
#             box center position only.

#         Returns
#         -------
#         ndarray
#             Returns an array of length N, where the i-th element contains the
#             squared Mahalanobis distance between (mean, covariance) and
#             `measurements[i]`.

#         """
#         mean, covariance = self.project(mean, covariance)
#         if only_position:
#             mean, covariance = mean[:2], covariance[:2, :2]
#             measurements = measurements[:, :2]

#         cholesky_factor = np.linalg.cholesky(covariance)
#         d = measurements - mean
#         z = scipy.linalg.solve_triangular(
#             cholesky_factor, d.T, lower=True, check_finite=False,
#             overwrite_b=True)
#         squared_maha = np.sum(z * z, axis=0)
#         return squared_maha
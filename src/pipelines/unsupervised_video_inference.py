import os 
import json
import pdb


from datetime import datetime
from tqdm import tqdm

from src.data_loader import get_loader
from src.extractor import get_detector, get_tracker, get_extractor

from src.aggregator import get_aggregator
from src.ucp import get_ucp
from moviepy.editor import VideoFileClip
from src.utils import setup_logger



class UnsupervisedVideoInference():
    def __init__(self, config) -> None:
        self.config = config
    
    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get all essential components
        video_loader = get_loader(self.config)
        face_detector = get_detector(self.config)
        tracker = get_tracker(self.config)
        emot_extractor = get_extractor(self.config)
        aggregator = get_aggregator(self.config)
        ucp = get_ucp(self.config)

        video_data = video_loader.get_list_item()

        for (f_p_in, f_p_out, start_second) in tqdm(video_data): # start second for each video segment in a large video
            # Recording processing time
            start = datetime.now()

            # read video
            video = VideoFileClip(f_p_in)
            video_num_frames = int(video.fps * video.duration)
            video_fps = int(video.fps)
            
            # all-in-one process: detecting, tracking, extracking 
            '''
                All-in-one process:
                    + Extracting and detecting faces while performing tracking
            '''
            video_es_signals, video_es_offset, _ = tracker.run(video, face_detector, emot_extractor)
            

            exist_signal = (len(video_es_signals) != 0)
            exist_cp = False

            # detecting individual change points and ucp
            res_stat = []
            res_cp = []
            res_score = []
            if exist_signal:
                all_peaks_track, _, all_scores_sm_track = ucp.run(video_es_signals)

                exist_cp = (len(all_peaks_track) != 0)

                if exist_cp:
                    all_refined_peaks_track = []
                    all_refined_scores_sm_track = []

                    # refine peak
                    for idx in range(len(all_peaks_track)):
                        peak_track = all_peaks_track[idx]
                        sm_score_track = all_scores_sm_track[idx]

                        if peak_track is None:
                            continue

                        start_offset_track = video_es_offset[idx][0]
                        refined_start_offset_track = peak_track + start_offset_track

                        all_refined_peaks_track.append(refined_start_offset_track)
                        all_refined_scores_sm_track.append(sm_score_track)

                    # aggregate to find final change point
                    res_cp, res_score, res_stat, individual_cp = aggregator.run(all_peaks_track, all_refined_scores_sm_track,
                                                                                video_num_frames, video_fps, start_second)

            time_processing = datetime.now() - start
    
            result = {"final_cp_result": res_cp, 
                        "final_cp_llr": res_score,
                        "type": "video", 
                        "input_path": f_p_in,
                        "total_video_frame": video_num_frames, 
                        "num_frame_skip": self.config.tracker.skip_frame,
                        'time_processing': int(time_processing.total_seconds()),
                        "fps": int(video_fps), 
                        "individual_cp_result": individual_cp,
                        "stat_segment_seconds_total_cp_accum": res_stat
                        }
            
            with open(f_p_out, 'w') as fp:
                json.dump(result, fp, indent=4)




            
            







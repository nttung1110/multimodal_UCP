from .base_aggregator import *
from src.utils import Config

import numpy as np
import pdb

class VideoCPAggregator(BaseCPAggregator):
    def __init__(self, config: Config):
        super(VideoCPAggregator, self).__init__(config)
        self.config = config

        # init model

    def _segmenting_video(self, video_num_frames):
        # self.num_segments # num_intervals

        sequence_length = int(video_num_frames / self.config.num_segments)

        if sequence_length == 1:
            return [0, self.config.num_segments, video_num_frames]
        if sequence_length == 0:
            return [0, video_num_frames]

        list_index = []
        start = 0
        while True:
            start += sequence_length - 1
            if start >= video_num_frames:
                list_index.append(video_num_frames)
                break
            else:
                list_index.append(start)

        return list_index
    
    def _aggregate_cp(self, segment_ids, binary_matrix, score_matrix, fps):
        h = [0]*(len(segment_ids))
        score = [0]*(len(segment_ids))

        for i in range(len(h)):
            h[i] = np.sum(binary_matrix[:, 0:(segment_ids[i])])
            score[i] = np.sum(score_matrix[:, 0:(segment_ids[i])])

        # calculate f(M) based on g
        index_list = []
        total_change_point_list = []
        total_score_list = []

        for i in range(len(h)):
            if i == 0:
                total_change_point_list.append(int(h[0]))
                index_list.append(segment_ids[0])
                total_score_list.append(score[0])
                continue

            g = h[i] - h[i-1]
            sum_score = score[i] - score[i-1]

            total_change_point_list.append(int(g))
            total_score_list.append(float(sum_score/g))#normalize based on total of change point
            index_list.append(segment_ids[i])

        # sort with respect to total_change_point_list
        idx_sort = np.argsort(np.array(total_change_point_list))
        prioritized_index_list = [index_list[a]  for a in idx_sort]
        prioritized_score_list = [total_score_list[a] for a in idx_sort]

        # this index refers to the frame index

        # view as list
        if len(prioritized_index_list) >= self.config.max_cp:
            max_index_list = prioritized_index_list[-self.config.max_cp:]
            max_score_list = prioritized_score_list[-self.config.max_cp:]
        else:
            max_index_list = prioritized_index_list
            max_score_list = prioritized_score_list

        final_res = []
        for res in max_index_list:
            convert_seconds = int(res/fps)
            final_res.append(convert_seconds)

        return final_res, max_score_list, total_change_point_list


    def run(self, all_refined_peaks_track, all_refined_scores_sm_track,
            video_num_frames, fps, start_second):
        # convert refined peak indices to binary matrix
        binary_cp_matrix = np.zeros((len(all_refined_peaks_track), video_num_frames))
        score_cp_matrix = np.zeros((len(all_refined_peaks_track), video_num_frames))

        # num track x num frames
        for idx_track, each_track in enumerate(all_refined_peaks_track):
            # print(all_refined_peaks_track)
            for i, each_cp_index in enumerate(each_track):
                try:
                    binary_cp_matrix[idx_track][each_cp_index] = 1
                except:
                    continue
                score_cp_matrix[idx_track][each_cp_index] = all_refined_scores_sm_track[idx_track][i]
        
        res_segment_ids = self._segmenting_video(video_num_frames)


        res_cp, res_score, stat_total_cp_interval = self._aggregate_cp(res_segment_ids, binary_cp_matrix, score_cp_matrix, fps)

        # Shift the final set of change points (res_cp) to align with the original video time indexes.
        for idx in range(len(res_cp)):
            res_cp[idx] += start_second

        individual_cp = [(a/fps).astype(int).tolist() for a in all_refined_peaks_track]

        # convert stat infor to second-based
        res_stat = []

        for a, b in zip(res_segment_ids, stat_total_cp_interval):
            a_second = int(a/fps)
            res_stat.append((a_second, b))

        return res_cp, res_score, res_stat, individual_cp

    
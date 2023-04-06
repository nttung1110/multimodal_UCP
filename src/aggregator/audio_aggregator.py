from src.utils import Config
from .base_aggregator import BaseCPAggregator

class AudioCPAggregator(BaseCPAggregator):
    def __init__(self, config: Config):
        self.config = config
        # init model, pass for the inheritance class

    
    def run(self, 
            all_peaks_cp_track, 
            offset_signals, 
            all_scores_pick_softmax_track):

        final_cp = []
        final_score = []

        for cp_list, score_list, each_offset in zip(all_peaks_cp_track, all_scores_pick_softmax_track, offset_signals):
            for cp, score in zip(cp_list, score_list):
                if cp not in final_cp:
                    final_cp.append(each_offset[1] + 96*cp/1000) # start + 96*cp/1000
                    final_score.append(score)

        # find $max_cp most significant change point
        if len(final_cp) >= self.config.max_cp:
            final_cp = final_cp[:self.config.max_cp]
            final_score = final_score[:self.config.max_cp]

        final_cp_res = [int(a) for a in list(final_cp)]

        return final_cp_res, final_score
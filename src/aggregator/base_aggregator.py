from src.utils import Config

class BaseCPAggregator:
    def __init__(self, config: Config):
        self.config = config
        # init model, pass for the inheritance class

    def run(self, all_peaks_cp_track, all_scores_pick_softmax_track):
        all_cp = []

        final_cp = []
        final_score = []

        for cp_list, score_list in zip(all_peaks_cp_track, all_scores_pick_softmax_track):
            for cp, score in zip(cp_list, score_list):
                if cp not in final_cp:
                    final_cp.append(int(cp))
                    final_score.append(score)

        # find $max_cp most significant change point
        if len(final_cp) >= self.config.max_cp:
            final_cp = final_cp[:self.config.max_cp]
            final_score = final_score[:self.config.max_cp]

        return final_cp, final_score
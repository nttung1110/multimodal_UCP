from src.utils import Config
import roerich
import pdb
import torch
import matplotlib.pyplot as plt
import numpy as np

# from utils.visualise import display_signal
softmax = torch.nn.Softmax(dim=1)

from roerich.algorithms import OnlineNNRuLSIF

class BaseUCP:
    def __init__(self, config: Config):
        self.config = config
        # init model, pass for the inheritance class

    def detect_cp(self, data):

        cpd = OnlineNNRuLSIF(net=self.config.net, scaler=self.config.scaler, metric=self.config.metric, 
                            periods=self.config.periods, window_size=self.config.window_size,
                            lag_size=self.config.lag_size, step=self.config.step, 
                            n_epochs=self.config.n_epochs, lr=self.config.lr, lam=self.config.lam, 
                            optimizer=self.config.optimizer, alpha=self.config.alpha)
        
        scores, peaks = cpd.predict(data)

        return scores, peaks

    def run(self, es_signals):
        # se_track is only apply for video to keep track of the offset of track
        print("========Detecting change point from individual ES track===========")

        all_scores_cp_track = []
        all_peaks_cp_track = []
        all_scores_sm_cp_track = []

        for each_signal in es_signals:
            if each_signal.shape[0] == 0:
                continue
            res_scores_track, res_peaks_track = self.detect_cp(each_signal)

            if len(res_peaks_track) == 0:
                # no cp exist
                res_scores_track = None
                res_peaks_track = None
                sm = None

            else:
                score_pick_track = []
                for idx, each_cp in enumerate(res_peaks_track):
                    score_pick_track.append(res_scores_track[each_cp])

                sm = softmax(torch.Tensor(np.array([score_pick_track])))[0].tolist()

            all_scores_cp_track.append(res_scores_track)
            all_peaks_cp_track.append(res_peaks_track)
            all_scores_sm_cp_track.append(sm)

        return all_peaks_cp_track, all_scores_cp_track, all_scores_sm_cp_track
    
    # this is temporary, create a new text_ucp class if needed
    def run_for_text(self, es_signals, start_offset_utt_by_speaker):
        print("========Detecting change point from individual ES track===========")
        
        s_offset = [start_offset_utt_by_speaker['s1'],
                    start_offset_utt_by_speaker['s2']]

        all_peaks_track_refined = []
        all_scores_pick_softmax_track = []

        for each_signal, each_offset in es_signals:
            if each_signal.shape[0] == None:
                continue

            res_scores_track, res_peaks_track = self.detect_cp(each_signal)

            if len(res_peaks_track) == 0:
                # no cp exist
                continue

            score_pick_track = []
            for idx, each_cp in enumerate(res_peaks_track):
                score_pick_track.append(res_scores_track[each_cp])

            sm = softmax(torch.Tensor(np.array([score_pick_track])))[0].tolist()



        if len(res_peaks_track_s1) != 0:
            # having cp
            score_pick_track = []
            refined_peaks_track_s1 = [] # cp location should be character level
            offset_s1 = start_offset_utt_by_speaker['s1']

            for idx, each_cp in enumerate(res_peaks_track_s1):
                refined_peaks_track_s1.append(offset_s1[each_cp-1]) # convert utterance level index to character level index 
                score_pick_track.append(res_scores_track_s1[each_cp])

            score_sm_cp_s1 = softmax(torch.Tensor(np.array([score_pick_track])))[0].tolist()

        # check condition before doing s2
        if s2_signal.shape[0] != 0:

            all_peaks_track_refined = [refined_peaks_track_s1,
                                        refined_peaks_track_s2]
            all_scores_pick_softmax_track = [score_sm_cp_s1,
                                            score_sm_cp_s2]
        else:
            # s1 only
            all_peaks_track_refined = [refined_peaks_track_s1]
            all_scores_pick_softmax_track = [score_sm_cp_s1]

    
    
import os 
import json
import pdb


from datetime import datetime
from src.extractor import get_extractor
from src.aggregator import get_aggregator
from src.ucp import get_ucp
from src.data_loader import get_loader
from tqdm import tqdm

from src.utils import setup_logger



class UnsupervisedAudioInference():
    def __init__(self, config) -> None:
        self.config = config
    
    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get audio loader
        audio_loader = get_loader(self.config)
        
        # get ES extractor
        extractor = get_extractor(self.config)

        # get aggregator
        aggregator = get_aggregator(self.config)

        # get ucp
        ucp = get_ucp(self.config)

        # get data
        audio_data = audio_loader.get_list_item()

        for (f_p_in, f_p_out) in tqdm(audio_data):
            # Recording processing time
            start = datetime.now()

            # extract features
            audio_es, offset, l_in_sec, _ = extractor.run(f_p_in)

            
            if len(audio_es) == 0:
                # no voice track found => no change point
                final_cp_res = []
                res_score = []
                individual_cp = []
                length_audio = l_in_sec

            else:
                # detect with ucp
                all_peaks_track, _, all_scores_sm_track = ucp.run(audio_es)

                # aggregating change point
                final_cp_res, res_score = aggregator.run(all_peaks_track, offset, all_scores_sm_track)

                individual_cp = [a.astype(int).tolist() for a in all_peaks_track]

            # save output 
            time_processing = datetime.now() - start
            res = {
                    'final_cp_result': final_cp_res,
                    'final_cp_llr': res_score,
                    'type': 'audio',
                    'time_processing': int(time_processing.total_seconds()),
                    'individual_cp': individual_cp,
                    'length_audio': l_in_sec
                }
        
            with open(f_p_out, 'w') as fp:
                json.dump(res, fp, indent=4)




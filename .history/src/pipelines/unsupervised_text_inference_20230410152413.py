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



class UnsupervisedTextInference():
    def __init__(self, config) -> None:
        self.config = config
    
    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get aggregator
        aggregator = get_aggregator(self.config)

        # get ucp
        ucp = get_ucp(self.config)

        # get text loader
        text_loader = get_loader(self.config)

        # get extractor
        extractor = get_extractor(self.config)

        text_data = text_loader.get_list_item()
        
        for (f_p_in, f_p_out) in tqdm(text_data): 
            # if 'en' in f_p_in:
            #     continue # only read chinese version
            print(f_p_in)
            if os.path.exists(f_p_out):
                print(f"Skipping: {f_p_out}, results are available")
                continue
            # Recording processing time
            start = datetime.now()
            
            # extract features
            text_es, _ = extractor.run(f_p_in)

            if len(text_es) == 0:
                # no text track found => no change point
                final_cp_res = []
                res_score = []
                individual_cp = []
            else:
                # detect with ucp
                all_peaks_track, _, all_scores_sm_track = ucp.run(text_es)

                # refined peak track and convert it back to character level index
                all_peaks_track_refined = []
                all_scores_pick_softmax_track = []

                tmp = {'s1': all_peaks_track[0], 's2': all_peaks_track[1]}
                tmp_score = {'s1': all_scores_sm_track[0], 's2': all_scores_track[1]}

                # aggregating change point
                final_cp_res, res_score = aggregator.run(all_peaks_track, all_scores_sm_track)

                individual_cp = [a.astype(int).tolist() for a in all_peaks_track]

            # save output 
            time_processing = datetime.now() - start
            res = {
                    'final_cp_result': list(final_cp_res),
                    'final_cp_llr': res_score,
                    'type': 'text',
                    'time_processing': int(time_processing.total_seconds()),
                    'individual_cp': list(individual_cp)
                }
            with open(f_p_out, 'w') as fp:
                json.dump(res, fp, indent=4)


        
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
            text_es, text_offset = extractor.run(f_p_in)

            if len(text_es) == 0:
                # no text track found => no change point
                final_cp_res = []
                res_score = []
                individual_cp = []
            else:
                # detect with ucp
                all_peaks_track, _, all_scores_sm_track = ucp.run_for_text(text_es, text_offset)

                # aggregating change point
                final_cp_res, res_score = aggregator.run(all_peaks_track, all_scores_sm_track)

                individual_cp = all_peaks_track

            # save output 
            time_processing = datetime.now() - start
            res = {
                    'final_cp_result': list(final_cp_res),
                    'final_cp_llr': res_score,
                    'type': 'text',
                    'time_processing': int(time_processing.total_seconds()),
                    'individual_cp': all_peaks_track
                }
            with open(f_p_out, 'w') as fp:
                json.dump(res, fp, indent=4)


        
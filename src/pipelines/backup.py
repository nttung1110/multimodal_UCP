# import os 
# import numpy as np 
# import pdb
# import torch
# import json
# import os.path as osp
# import pdb
# import tensorflow_hub as hub
# import tensorflow as tf


# from dotmap import DotMap
# from pydub import AudioSegment 
# from pyannote.audio import Audio 
# from pyannote.audio import Pipeline
# from datetime import datetime
# from pydub import AudioSegment
# from tqdm import tqdm




# # temporary import
# import sys
# sys.path.append('./ES_extractor')

# from ES_extractor.audio_feat import AudioES
# from UCP.inference_ucp import detect_CP_tracks
# from CP_aggregator import aggregator_core

# # from CP_aggregator.segment_core import UniformSegmentator
# # from CP_aggregator.aggregator_core import SimpleAggregator


# def run_pipeline_single_audio(args, path_audio_file, audio_es_module, path_write_res):

#     # get length of audio first
#     audio = AudioSegment.from_file(path_audio_file)
#     length = int(audio.duration_seconds)


#     start = datetime.now()
#     # extract features by speakers
#     audio_es_signals, offset_signals, _ = audio_es_module.extract_audio_track(path_audio_file)

#     no_cp_confirm = False

#     if len(audio_es_signals) == 0:
#         # no voice track found => no change point
#         no_cp_confirm = True
#         final_cp = []
#         res_score = []
#         individual_cp = []
#         length_audio = length

#     else:
#         # UCP Detector
#         all_peaks_track, all_scores_track = detect_CP_tracks(audio_es_signals)
        
#         # perform softmax on score
#         softmax = torch.nn.Softmax(dim=1)
#         all_scores_pick_softmax_track = []

#         for each_peak_track, each_score_track in zip(all_peaks_track, all_scores_track):
#             score_pick_track = []

#             for idx, each_cp in enumerate(each_peak_track):
#                 score_pick_track.append(each_score_track[each_cp])

#             sm = softmax(torch.Tensor(np.array([score_pick_track])))

#             all_scores_pick_softmax_track.append(sm[0].tolist())

#         # Aggregate to find final change point

#         individual_cp = [a.astype(int).tolist() for a in all_peaks_track]

#         final_cp, res_score = aggregator_core.simple_aggregator(all_peaks_track, offset_signals, all_scores_pick_softmax_track, args.max_cp)
#         final_cp_res = [int(a) for a in list(final_cp)]

#     time_processing = datetime.now() - start
#     res = {'final_cp_result': final_cp_res,
#             'final_cp_llr': res_score,
#             'type': 'audio',
#             'time_processing': int(time_processing.total_seconds()),
#             'individual_cp': individual_cp,
#             'length_audio': length}

#     write_fname = file_name.split('.')[0]+'.json'
#     with open(path_write_res, 'w') as fp:
#         json.dump(res, fp, indent=4)

# if __name__ == "__main__":
#     # init argument
#     args = DotMap()
#     args.min_length_audio_signal = 20
#     args.max_cp = 3


#     # path_inference_audio_path = "/home/nttung/research/Monash_CCU/mini_eval/audio_data/DARPA_wav_from_video"

#     # official CCU data path
#     path_inference_audio_path = '/home/nttung/research/Monash_CCU/mini_eval/sub_data/converted_audio'

#     path_write_out_path = "/home/nttung/research/Monash_CCU/mini_eval/audio_module/AUDIO_CCU_output_v1"

#     if osp.isdir(path_write_out_path) is False:
#         os.mkdir(path_write_out_path)

#     # initilize audio ES model
#     audio_es_module = AudioES(args)
    
#     for file_name in tqdm(os.listdir(path_inference_audio_path)):
#         print(file_name)

#         full_path_audio = osp.join(path_inference_audio_path, file_name)

#         path_write_res = osp.join(path_write_out_path, file_name.split('.')[0]+'.json')
        

#         run_pipeline_single_audio(args, full_path_audio, audio_es_module, path_write_res)  

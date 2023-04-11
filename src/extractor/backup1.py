# # """
# # # for speechbrain
# # !pip install -qq torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 torchtext==0.12.0
# # !pip install -qq speechbrain==0.5.12

# # # pyannote.audio
# # !pip install -qq pyannote.audio

# # # for visualization purposes
# # !pip install -qq ipython==7.34.0

# # # for audio slicing
# # !pip install pydub
# # """

# # list_rep = [a1, a2, ..., am]
# # list_offset = [(s1, e1), (s2, e2), ..., (sm, em)]

# # a1.shape = (d, l1)
# # a2.shape = (d, l2)


# import os
# import tensorflow_hub as hub
# import numpy as np
# import pdb
# import tensorflow as tf


# from scipy.io import wavfile
# from pydub import AudioSegment 
# from pyannote.audio import Audio 
# from pyannote.audio import Pipeline

# def audio_slicing(newAudio, start, stop):
#     #Works in milliseconds
#     """
#     @params: start stop is float in sec -> need to transform to minisec
#     @return: a part of original audio specified by start and stop time
#     """
#     start = int(start * 1000)  
#     stop = int(stop * 1000) 
#     #print("length of trunk:", stop/1000 - start/1000,  (stop-start)//96)
#     newAudio = newAudio[start : stop]
#     return newAudio

# # `wav_as_float_or_int16` can be a numpy array or tf.Tensor of float type or
# # int16. The sample rate must be 16kHz. Resample to this sample rate, if
# # necessary.
# def emotion_signals(module, audio, start, stop):
#     """
#     @params: start, stop in seconds
#     @return: a li * d embeddings in which li represents for duration of the audio, while d is emotion signals representation dimension
#     """
    
#     wav = audio_slicing(audio, start, stop).set_frame_rate(16000)
#     wav_as_np = np.array(wav.get_array_of_samples())
#     emb_dict = module(samples=wav_as_np, sample_rate=16000)
#     # For a description of the difference between the two endpoints, please see our
#     # paper (https://arxiv.org/abs/2002.12764), section "Neural Network Layer".
#     emb = emb_dict['embedding']
#     #emb_layer19 = emb_dict['layer19']
#     # Embeddings are a [time, feature_dim] Tensors.
#     emb.shape.assert_is_compatible_with([None, 512])
#     #emb_layer19.shape.assert_is_compatible_with([None, 12288])
#     #print(emb.shape[0])
#     return tf.transpose(emb)

# def extract_audio_track(path_file, module, pipeline):
#     """
#     @params: awv file directory
#     list_rep[m]: list_rep[i] has dimension li * d with d (=12288) is represents for audio emotion features, and l_i represents for time series of track i
#     list_offset[m]: list_offset[i] = (start, stop) information for track i
#     length[m]: duration of each track 
#     """
#     #Read file
#     filename, ext = os.path.splitext(path_file)
#     if (ext != '.wav'):
#         print("The function needs wav file as input", ext)
#         return 1
#     audio_name = filename.split('\\')[-1]
    
#     #Speaker diarization pipeline
#     diarization = pipeline(f"{filename}.wav")
#     list_rep, list_offset, length = [], [], []
    
#     #Extract start, stop, and duration of each track
#     for turn, _, speaker in diarization.itertracks(yield_label=True):
#         list_offset.append([speaker, turn.start, turn.end])
#     list_offset = sorted(list_offset)
#     length = [y - x for _, x, y in list_offset]
#     #print(len(list_offset))
#     audio = AudioSegment.from_file(path_file) 
#     for _, start, stop in list_offset:
#         if (start == None or stop == None or int(stop * 1000) - int(start * 1000) <= 1000):
#           continue
#         list_rep.append(emotion_signals(module, audio, start, stop))
    
#     return list_rep, list_offset, length

# class AudioES():
#     def __init__(self, args):
#         self.args = args
#         self.init_model()

    
#     def init_model(self):
#         print("========Initializing model===========")

#         os.environ["TFHUB_CACHE_DIR"] = "../tmp"
#         self.module = hub.load('https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/2')
#         self.pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token="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")

#     def audio_slicing(self, newAudio, start, stop):
#     #Works in milliseconds
#         """
#         @params: start stop is float in sec -> need to transform to minisec
#         @return: a part of original audio specified by start and stop time
#         """
#         start = int(start * 1000)  
#         stop = int(stop * 1000) 
#         #print("length of trunk:", stop/1000 - start/1000,  (stop-start)//96)
#         newAudio = newAudio[start : stop]
#         return newAudio

#         # `wav_as_float_or_int16` can be a numpy array or tf.Tensor of float type or
#         # int16. The sample rate must be 16kHz. Resample to this sample rate, if
#         # necessary.

#     def emotion_signals(self, module, audio, start, stop):
#         """
#         @params: start, stop in seconds
#         @return: a li * d embeddings in which li represents for duration of the audio, while d is emotion signals representation dimension
#         """
        
#         wav = audio_slicing(audio, start, stop).set_frame_rate(16000)
#         wav_as_np = np.array(wav.get_array_of_samples())
#         #print(wav_as_np.shape)
#         emb_dict = module(samples=wav_as_np, sample_rate=16000)
#         # For a description of the difference between the two endpoints, please see our
#         # paper (https://arxiv.org/abs/2002.12764), section "Neural Network Layer".
#         emb = emb_dict['embedding']
#         #emb_layer19 = emb_dict['layer19']
#         # Embeddings are a [time, feature_dim] Tensors.
#         emb.shape.assert_is_compatible_with([None, 512])
#         #emb_layer19.shape.assert_is_compatible_with([None, 12288])
#         #print(emb.shape[0])
#         return tf.transpose(emb)


#     def extract_audio_track(self, path_file):
#         """
#         @params: awv file directory
#         list_rep[m]: list_rep[i] has dimension d*li with d (=512) is represents for audio emotion features, and l_i represents for time series of track i
#         list_offset[m]: list_offset[i] = (start, stop) information for track i
#         length[m]: duration of each track 
#         """
#         #Read file
#         filename, ext = os.path.splitext(path_file)
#         if (ext != '.wav'):
#             print("The function needs wav file as input", ext)
#             return 1
#         audio_name = filename.split('\\')[-1]
        
#         #Speaker diarization pipeline
#         diarization = self.pipeline(f"{filename}.wav")
#         list_rep, list_offset, length = [], [], []

#         #Extract start, stop, and duration of each track
#         for turn, _, speaker in diarization.itertracks(yield_label=True):
#             list_offset.append([speaker, turn.start, turn.end])
#         list_offset = sorted(list_offset)
#         length = [y - x for _, x, y in list_offset]
#         #print(len(list_offset))
#         audio = AudioSegment.from_file(path_file) 
#         for _, start, stop in list_offset:
#             if (start == None or stop == None or int(stop * 1000) - int(start * 1000) <= 1000):
#                 continue
#             list_rep.append(self.emotion_signals(self.module, audio, start, stop))
        
#         # #Extract start, stop, and duration of each track
#         # for turn, _, speaker in diarization.itertracks(yield_label=True):
#         #     list_offset.append([speaker, turn.start, turn.end])
#         # list_offset = sorted(list_offset)
#         # length = [y - x for _, x, y in list_offset]

#         # for _, start, stop in list_offset:
#         #     list_rep.append(self.emotion_signals(self.module, path_file, start, stop))
#         #print(len(list_rep))

#         # detach cpu and filter those short signals
#         audio_es_signals = []
#         corres_offset = []
            
#         for each_rep, each_offset in zip(list_rep, list_offset):
#             a = each_rep.cpu().numpy()
#             a = np.transpose(a)

#             len_signal = a.shape[0]
#             if len_signal < self.args.min_length_audio_signal:
#                 continue

#             audio_es_signals.append(a)
#             corres_offset.append(each_offset)


#         return audio_es_signals, corres_offset, length


# if __name__ == "__main__":
#     # tf.enable_v2_behavior()
#     # assert tf.executing_eagerly()

#     # set this as ENVIRONMENT VARIABLE: export TFHUB_CACHE_DIR=./tmp


#     # disabling gpu
#     # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#     module = hub.load('https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/2')

#     pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token="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")


#     #unit test
#     path_audio = '../../audio_data/DARPA_wav_from_video/M01000AJ9_0.wav'
#     audio_feat_track = extract_audio_track(path_audio, module, pipeline)
#     pdb.set_trace()

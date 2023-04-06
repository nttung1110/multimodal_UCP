from src.utils import Config

import tensorflow_hub as hub
import numpy as np
import pdb
import torch
import os
import tensorflow as tf


from scipy.io import wavfile
from pydub import AudioSegment 
from pyannote.audio import Pipeline as Pipe_Diar
from pyannote.audio import Audio 

class BaseExtractor:
    def __init__(self, config: Config):
        self.config = config
        # init model, pass for the inheritance class

    def _init_model(self):
        print("=========Loading diarization model==============")
        self.pipeline = Pipe_Diar.from_pretrained('pyannote/speaker-diarization',
                                                use_auth_token=self.config.auth_token)
    
    def _diarize(self, path_file):
        #Read file
        filename, ext = os.path.splitext(path_file)
        if (ext != '.wav'):
            print("The function needs wav file as input", ext)
            return 1
        audio_name = filename.split('\\')[-1]
        
        #Speaker diarization pipeline
        diarization = self.pipeline(f"{filename}.wav")
        list_offset, length = [], []

        #Extract start, stop, and duration of each track
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            list_offset.append([speaker, turn.start, turn.end])

        list_offset = sorted(list_offset)
        length = [y - x for _, x, y in list_offset]

        return list_offset, length
    
    



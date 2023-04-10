from .base_audio_extractor import *
from src.utils import Config

import librosa

class SpectrogramExtractor(BaseExtractor):
    def __init__(self, config: Config):
        super(SpectrogramExtractor, self).__init__(config)
        self.config = config

        # init model
        self._init_model()

    def _feat_signals(self, audio, start, stop, sample_rate):
        audio_slice = audio[int(start*sample_rate):int(stop*sample_rate)]

        spectrogram = librosa.feature.melspectrogram(y=audio_slice, sr=sample_rate, n_fft=2048, hop_length=512)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        return spectrogram_db

    def run(self, path_file):
        # diarization step
        list_offset, length = self._diarize(path_file)

        # extract feature step
        audio_signal, sample_rate = librosa.load(path_file)
        list_rep = []

        for _, start, stop in list_offset:
            if (start == None or stop == None or int(stop * 1000) - int(start * 1000) <= 1000):
                continue
            list_rep.append(self._feat_signals(audio_signal, start, stop, sample_rate))
        

        audio_es_signals = []
        corres_offset = []

        for each_rep, each_offset in zip(list_rep, list_offset):
            a = each_rep # dump
                        
            audio_es_signals.append(a)
            corres_offset.append(each_offset)

        length_in_sec = int(len(audio_signal)/sample_rate)

        return audio_es_signals, corres_offset, length_in_sec, length
        
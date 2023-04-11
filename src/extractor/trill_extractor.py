from .base_audio_extractor import *
from tqdm import tqdm

from src.utils import Config

class TrillExtractor(BaseExtractor):
    def __init__(self, config: Config):
        super(TrillExtractor, self).__init__(config)
        self.config = config

        # init model
        self._init_model()

    def _audio_slicing(self, newAudio, start, stop):
        #Works in milliseconds
        """
        @params: start stop is float in sec -> need to transform to minisec
        @return: a part of original audio specified by start and stop time
        """
        start = int(start * 1000)  
        stop = int(stop * 1000) 
        #print("length of trunk:", stop/1000 - start/1000,  (stop-start)//96)
        newAudio = newAudio[start : stop]
        return newAudio

        # `wav_as_float_or_int16` can be a numpy array or tf.Tensor of float type or
        # int16. The sample rate must be 16kHz. Resample to this sample rate, if
        # necessary.

    def _init_model(self):
        # init diarization
        super(TrillExtractor, self)._init_model()

        os.environ["TFHUB_CACHE_DIR"] = "./tmp"
        print("========Loading Trill Audio model==============")
        pdb.set_trace()
        # gpu = tf.config.experimental.list_physical_devices('GPU')[0]
        # tf.config.experimental.set_memory_growth(gpu, True)
        # tf.config.set_visible_devices(gpu, 'GPU')
        
        with tf.device('/device:GPU:1'):
            self.module = hub.load(self.config.path_url_trill)
        # self.module = hub.load(self.config.path_url_trill)
            

    def _feat_signals(self, audio, start, stop):
        """
        @params: start, stop in seconds
        @return: a li * d embeddings in which li represents for duration of the audio, while d is emotion signals representation dimension
        """
        
        wav = self._audio_slicing(audio, start, stop).set_frame_rate(self.config.frame_rate)
        wav_as_np = np.array(wav.get_array_of_samples())
        #print(wav_as_np.shape)
        emb_dict = self.module(samples=wav_as_np, sample_rate=self.config.sample_rate)
        # For a description of the difference between the two endpoints, please see our
        # paper (https://arxiv.org/abs/2002.12764), section "Neural Network Layer".
        emb = emb_dict['embedding']
        #emb_layer19 = emb_dict['layer19']
        # Embeddings are a [time, feature_dim] Tensors.
        emb.shape.assert_is_compatible_with([None, 512])
        #emb_layer19.shape.assert_is_compatible_with([None, 12288])
        #print(emb.shape[0])
        return tf.transpose(emb)
    
    def run(self, path_file):
        """
        @params: awv file directory
        list_rep[m]: list_rep[i] has dimension d*li with d (=512) is represents for audio emotion features, and l_i represents for time series of track i
        list_offset[m]: list_offset[i] = (start, stop) information for track i
        length[m]: duration of each track 
        """

        # diarization step
        list_offset, length = self._diarize(path_file)

        # extract emotion step
        audio = AudioSegment.from_file(path_file) 
        list_rep = []
        for _, start, stop in list_offset:
            if (start == None or stop == None or int(stop * 1000) - int(start * 1000) <= 1000):
                continue
            list_rep.append(self._feat_signals(audio, start, stop))
        
        audio_es_signals = []
        corres_offset = []
            
        for each_rep, each_offset in zip(list_rep, list_offset):
            a = each_rep.cpu().numpy()
            a = np.transpose(a)

            len_signal = a.shape[0]
            if len_signal < self.config.min_length_audio_signal:
                continue

            audio_es_signals.append(a)
            corres_offset.append(each_offset)

        length_in_sec = int(audio.duration_seconds)

        return audio_es_signals, corres_offset, length_in_sec, length


    def run_both(self, path_file, ucp):
        """
            Different from run function: It perform both feature extraction and detecting
                CP at the same time for better inference time 
        """

        # disabling gpu 
        # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        # diarization step
        pdb.set_trace()
        list_offset, length = self._diarize(path_file)

        # extract emotion step
        audio = AudioSegment.from_file(path_file) 
        length_in_sec = int(audio.duration_seconds)

        all_scores_cp_track = []
        all_peaks_cp_track = []
        all_scores_sm_cp_track = []
        all_list_offset = []

        softmax = torch.nn.Softmax(dim=1)

        for a, start, stop in tqdm(list_offset):
            if (start == None or stop == None or int(stop * 1000) - int(start * 1000) <= 1000):
                continue
            try:
                feat = self._feat_signals(audio, start, stop)
            except:
                pdb.set_trace()
            feat = np.transpose(feat.cpu().numpy())
            
            len_signal = feat.shape[0]
            if len_signal < self.config.min_length_audio_signal:
                # ignore short tracks
                continue
            
            res_scores_track, res_peaks_track = ucp.detect_cp(feat)
            if len(res_peaks_track) == 0:
                # no cp exist
                continue
            else:
                score_pick_track = []
                for idx, each_cp in enumerate(res_peaks_track):
                    score_pick_track.append(res_scores_track[each_cp])

                sm = softmax(torch.Tensor(np.array([score_pick_track])))[0].tolist()

            all_scores_cp_track.append(res_scores_track)
            all_peaks_cp_track.append(res_peaks_track)
            all_scores_sm_cp_track.append(sm)
            all_list_offset.append((a, start, stop))
            
        return all_peaks_cp_track, all_scores_cp_track, \
                all_scores_sm_cp_track, all_list_offset, \
                length_in_sec

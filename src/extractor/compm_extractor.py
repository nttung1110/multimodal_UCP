
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import math
import pandas as pd
import pdb

from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model
from src.utils import Config
from googletrans import Translator
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence

from transformers import RobertaConfig, BertConfig

from tqdm import tqdm

class CoMPMExtractor:
    def __init__(self, config: Config):
        self.config = config
        # init model, pass for the inheritance class
        self._init_model()

    def _init_model(self):
        print("=========Loading CoMPM model==============")

        initial = self.config.initial
        model_type = self.config.pretrained
        freeze = self.config.freeze
        if freeze:
            freeze_type = 'freeze'
        else:
            freeze_type = 'no_freeze' 

        sample = self.config.sample
        if 'gpt2' in model_type:
            last = True
        else:
            last = False


        model_path = self.config.pretrained_model_ckpt
        clsNum = 7 # 7 emotion categories in daily dialog
        self.model = ERC_model(model_type, clsNum, last, freeze, initial)
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.cuda()    
        self.model.eval() 

        # translator
        self.translator = Translator()

        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    def padding(self, ids_list, tokenizer):
        max_len = 0
        for ids in ids_list:
            if len(ids) > max_len:
                max_len = len(ids)
        
        pad_ids = []
        for ids in ids_list:
            pad_len = max_len-len(ids)
            add_ids = [tokenizer.pad_token_id for _ in range(pad_len)]
            
            pad_ids.append(ids+add_ids)
        
        return torch.tensor(pad_ids)

    def encode_right_truncated(self, text, tokenizer, max_length=511):
        tokenized = tokenizer.tokenize(text)
        truncated = tokenized[-max_length:]    
        ids = tokenizer.convert_tokens_to_ids(truncated)
        
        return [tokenizer.cls_token_id] + ids

    def make_batch_roberta(self, sessions):
        batch_input, batch_labels, batch_speaker_tokens, batch_cur_utterance = [], [], [], []
        for session in sessions:
            data = session[0]
            label_list = session[1]
            
            context_speaker, context, emotion, sentiment = data
            now_speaker = context_speaker[-1]
            speaker_utt_list = []
            
            inputString = ""
            for turn, (speaker, utt) in enumerate(zip(context_speaker, context)):
                inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
                inputString += utt + " "
                
                if turn<len(context_speaker)-1 and speaker == now_speaker:
                    speaker_utt_list.append(self.encode_right_truncated(utt, self.roberta_tokenizer))
            
            concat_string = inputString.strip()
            batch_input.append(self.encode_right_truncated(concat_string, self.roberta_tokenizer))
            
            if len(label_list) > 3:
                label_ind = label_list.index(emotion)
            else:
                label_ind = label_list.index(sentiment)
            batch_labels.append(label_ind)        
            
            batch_speaker_tokens.append(self.padding(speaker_utt_list, self.roberta_tokenizer))
            batch_cur_utterance.append(context[-1])
        
        batch_input_tokens = self.padding(batch_input, self.roberta_tokenizer)
        batch_labels = torch.tensor(batch_labels)    
        
        return batch_input_tokens, batch_labels, batch_speaker_tokens, batch_cur_utterance

    def _preprocess_conversation(self, path_file):
        with open(path_file, "r") as fp:
            text_data = fp.read()

        text_data = text_data.split("\n")
        if text_data[-1] == "":
            text_data = text_data[:-1] # ignore last empty segment    

        # iterate and separate each segment
        s1 = [] # track utterance of speaker1 through conversation
        s2 = [] # track utterance of speaker2 through conversation

        # translate all text
        for idx, data in enumerate(text_data):
            # translate
            text_data[idx] = self.translator.translate(text_data[idx]).text

        # preprocess by eliminating auto-reply utterance

        # whoever is the first speaker will be assigned to s1
        first_speaker = None
        conversation_list = []

        running_offset = 0
        start_offset_s1 = []
        start_offset_s2 = []


        for segment in text_data:
            seg_id = int(segment.split(':')[0])
            seg_text = segment.split(':')[1]
            conversation_list.append(seg_text)

            if first_speaker == None:
                first_speaker = seg_id
                start_offset_s1.append(0)
                s1.append(seg_text)

                running_offset += len(seg_text)
                continue

            if seg_id == first_speaker:
                s1.append(seg_text)
                start_offset_s1.append(running_offset)
            else:
                s2.append(seg_text)
                start_offset_s2.append(running_offset)

            running_offset += len(seg_text)

        return {'s1': s1, 's2': s2}, text_data, {'s1': start_offset_s1, 's2': start_offset_s2}

    def run(self, path_file):
        utterance_by_speakers, conversation_list, start_offset_utt_by_speakers = self._preprocess_conversation(path_file)

        # prepare dataset loader
        dataset = Darpa_raw_loader(conversation_list, 'emotion')
        test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=self.make_batch_roberta)

        list_feat_s1 = []
        list_feat_s2 = []

        count = 0
        list_cur_utterance = {'s1': [], 's2': []}
        dict_feat = {}

        with torch.no_grad():
            for i_batch, data in enumerate(tqdm(test_dataloader)):
                """Prediction"""
                batch_input_tokens, _, batch_speaker_tokens, batch_utterance = data
                batch_input_tokens = batch_input_tokens.cuda()

                pred_logits = self.model(batch_input_tokens, batch_speaker_tokens) # (1, clsNum)

                feat_emotion = torch.nn.Softmax(dim=1)(pred_logits)

                # check whether this feat belongs to which speaker
                cur_utterance = batch_utterance[0]
                if cur_utterance not in dict_feat:
                    dict_feat[cur_utterance] = feat_emotion[0].tolist()

                # if cur_utterance in dict_speaker_utt['s1']:
                #     list_feat_s1.append(feat_emotion[0].tolist())
                #     list_cur_utterance['s1'].append(cur_utterance)
                # else:
                #     list_feat_s2.append(feat_emotion[0].tolist())
                #     list_cur_utterance['s2'].append(cur_utterance)
                
                count += 1

        for each_key_speaker in utterance_by_speakers:
            list_speaker_utterance = utterance_by_speakers[each_key_speaker]

            for each_utt in list_speaker_utterance:
                if each_key_speaker == 's1':
                    list_feat_s1.append(dict_feat[each_utt])
                else:
                    list_feat_s2.append(dict_feat[each_utt])

        feat_s1 = np.array(list_feat_s1)
        feat_s2 = np.array(list_feat_s2)

        all_es_text_feat_tracks = [feat_s1, feat_s2]
        return all_es_text_feat_tracks, start_offset_utt_by_speakers

# code from main compm model

class Darpa_raw_loader(Dataset):
    def __init__(self, conversation_list, dataclass):
        self.dialogs = []
        
        # f = open(txt_file, 'r')
        # dataset = f.readlines()
        # f.close()
        dataset = conversation_list
        
        temp_speakerList = []
        context = []
        context_speaker = []
        self.speakerNum = []      
        self.emoSet = set()
        self.sentiSet = set()
        # {'anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'}
        pos = ['happiness']
        neg = ['anger', 'disgust', 'fear', 'sadness']
        neu = ['neutral', 'surprise']
        emodict = {'anger': "anger", 'disgust': "disgust", 'fear': "fear", 'happiness': "happy", 'neutral': "neutral", 'sadness': "sad", 'surprise': "surprise"}
        self.sentidict = {'positive': pos, 'negative': neg, 'neutral': neu}
        for i, data in enumerate(dataset):
            if data == '\n' and len(self.dialogs) > 0:
                self.speakerNum.append(len(temp_speakerList))
                temp_speakerList = []
                context = []
                context_speaker = []
                continue
                
            # speaker = data.strip().split(':')[0]
            # utt = data.strip().split(':')[1]

            speaker = data[:6]
            utt = data[7:]
            # emo = data.strip().split('\t')[-1]
            
            # if emo in pos:
            #     senti = "positive"
            # elif emo in neg:
            #     senti = "negative"
            # elif emo in neu:
            #     senti = "neutral"
            # else:
            #     print('ERROR emotion&sentiment')       
            senti = 'None' # fake
            emo = 'None' 
            
            context.append(utt)
            if speaker not in temp_speakerList:
                temp_speakerList.append(speaker)
            speakerCLS = temp_speakerList.index(speaker)
            context_speaker.append(speakerCLS)
            
            self.dialogs.append([context_speaker[:], context[:], 'None', senti])
            self.emoSet.add('None')
        
        self.emoList = sorted(self.emoSet)   
        self.sentiList = sorted(self.sentiSet)
        self.labelList = self.emoList       
        self.speakerNum.append(len(temp_speakerList))
        
    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx], self.labelList, self.sentidict


class ERC_model(nn.Module):
    def __init__(self, model_type, clsNum, last, freeze, initial):
        super(ERC_model, self).__init__()
        self.gpu = True
        self.last = last
        
        """Model Setting"""
        # model_path = '/data/project/rw/rung/model/'+model_type
        model_path = model_type
        if 'roberta' in model_type:
            self.context_model = RobertaModel.from_pretrained(model_path)
                    
            if initial == 'scratch':
                config = RobertaConfig.from_pretrained(model_path)
                self.speaker_model = RobertaModel(config)
            else:
                self.speaker_model = RobertaModel.from_pretrained(model_path)
        elif model_type == 'bert-large-uncased':
            self.context_model = BertModel.from_pretrained(model_path)
            
            if initial == 'scratch':
                config = BertConfig.from_pretrained(model_path)
                self.speaker_model = BertModel(config)
            else:
                self.speaker_model = BertModel.from_pretrained(model_path)
        else:
            self.context_model = GPT2Model.from_pretrained(model_path)
            tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            tokenizer.add_special_tokens({'cls_token': '[CLS]', 'pad_token': '[PAD]'})
            self.context_model.resize_token_embeddings(len(tokenizer))
            
            self.speaker_model = GPT2Model.from_pretrained(model_path)
            self.speaker_model.resize_token_embeddings(len(tokenizer))
        self.hiddenDim = self.context_model.config.hidden_size
        
        zero = torch.empty(2, 1, self.hiddenDim).cuda()
        self.h0 = torch.zeros_like(zero) # (num_layers * num_directions, batch, hidden_size)
        self.speakerGRU = nn.GRU(self.hiddenDim, self.hiddenDim, 2, dropout=0.3) # (input, hidden, num_layer) (BERT_emb, BERT_emb, num_layer)
            
        """score"""
        # self.SC = nn.Linear(self.hiddenDim, self.hiddenDim)
        self.W = nn.Linear(self.hiddenDim, clsNum)
        
        """parameters"""
        self.train_params = list(self.context_model.parameters())+list(self.speakerGRU.parameters())+list(self.W.parameters()) # +list(self.SC.parameters())
        if not freeze:
            self.train_params += list(self.speaker_model.parameters())

    def forward(self, batch_input_tokens, batch_speaker_tokens):
        """
            batch_input_tokens: (batch, len)
            batch_speaker_tokens: [(speaker_utt_num, len), ..., ]
        """
        if self.last:
            batch_context_output = self.context_model(batch_input_tokens).last_hidden_state[:,-1,:] # (batch, 1024)
        else:
            batch_context_output = self.context_model(batch_input_tokens).last_hidden_state[:,0,:] # (batch, 1024)
        
        batch_speaker_output = []
        for speaker_tokens in batch_speaker_tokens:
            if speaker_tokens.shape[0] == 0:
                speaker_track_vector = torch.zeros(1, self.hiddenDim).cuda()
            else:
                if self.last:
                    speaker_output = self.speaker_model(speaker_tokens.cuda()).last_hidden_state[:,-1,:] # (speaker_utt_num, 1024)
                else:
                    speaker_output = self.speaker_model(speaker_tokens.cuda()).last_hidden_state[:,0,:] # (speaker_utt_num, 1024)
                speaker_output = speaker_output.unsqueeze(1) # (speaker_utt_num, 1, 1024)
                speaker_GRU_output, _ = self.speakerGRU(speaker_output, self.h0) # (speaker_utt_num, 1, 1024) <- (seq_len, batch, output_size)
                speaker_track_vector = speaker_GRU_output[-1,:,:] # (1, 1024)
            batch_speaker_output.append(speaker_track_vector)
        batch_speaker_output = torch.cat(batch_speaker_output, 0) # (batch, 1024)
                   
        final_output = batch_context_output + batch_speaker_output
        # final_output = batch_context_output + self.SC(batch_speaker_output)        
        context_logit = self.W(final_output) # (batch, clsNum)
        
        return context_logit



        
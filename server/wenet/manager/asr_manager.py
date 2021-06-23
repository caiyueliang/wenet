# -*-coding: utf-8 -*-
""""
-------------------------------------------------
   Description :  wenet server
   Author :       caiyueliang
   Date :         2021-06-17
-------------------------------------------------
"""
import os
import logging
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml
import numpy as np
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)


class ASRManager(object):
    def __init__(self, config):
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        with open(config, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)

        self.server_conf = configs['server']
        self.model_path = self.server_conf["model_path"]
        self.dict_path = self.server_conf["dict_path"]
        self.gpu_id = self.server_conf['gpu']
        self.train_yaml = self.server_conf['train_yaml']
        with open(self.train_yaml, 'r') as fin:
            model_configs = yaml.load(fin, Loader=yaml.FullLoader)

        self.recognize_conf = configs['recognize']
        self.mode = self.recognize_conf["mode"]
        self.decoding_chunk_size = self.recognize_conf["decoding_chunk_size"]
        self.num_decoding_left_chunks = self.recognize_conf["num_decoding_left_chunks"]
        self.ctc_weight = self.recognize_conf["ctc_weight"]
        self.beam_size = self.recognize_conf["beam_size"]
        self.simulate_streaming = self.recognize_conf["simulate_streaming"]

        self.feature_extraction_conf = configs['collate_conf']['feature_extraction_conf']

        # Load dict
        self.char_dict = dict()
        with open(self.dict_path, 'r') as fin:
            for line in fin:
                arr = line.strip().split()
                assert len(arr) == 2
                self.char_dict[int(arr[1])] = arr[0]
        self.eos = len(self.char_dict) - 1

        # Init asr model from configs
        self.model = init_asr_model(model_configs)
        load_checkpoint(self.model, self.model_path)
        # self.model = torch.jit.load(self.model_path)

        use_cuda = self.gpu_id >= 0 and torch.cuda.is_available()
        logging.info("[use_cuda] {}".format(use_cuda))
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.model = self.model.to(self.device)
        self.model.eval()

        logging.info("[ASRManager][__init__] success ... ")

    def load_data(self, wav_path):
        keys = []
        feats = []
        lengths = []
        try:
            waveform, sample_rate = torchaudio.load_wav(wav_path)
            mat = kaldi.fbank(
                waveform,
                num_mel_bins=self.feature_extraction_conf['mel_bins'],
                frame_length=self.feature_extraction_conf['frame_length'],
                frame_shift=self.feature_extraction_conf['frame_shift'],
                dither=0.0,
                energy_floor=0.0,
                sample_frequency=sample_rate
            )
            mat = mat.detach().numpy()
            feats.append(mat)
            keys.append(wav_path)
            lengths.append(waveform.shape[0])
        except (Exception) as e:
            print(e)
            logging.warning('read utterance {} error'.format(wav_path))
            pass

        xs_lengths = torch.from_numpy(np.array([x.shape[0] for x in feats], dtype=np.int32))
        xs_pad = torch.Tensor(feats)

        return keys, xs_pad, xs_lengths

    def recognize(self, wav_path):
        with torch.no_grad():
            keys, feats, feats_lengths = self.load_data(wav_path)
            # print("[key] {}".format(keys))
            # print("[feats] {}".format(feats))
            # print("[feats_lengths] {}".format(feats_lengths))
            feats = feats.to(self.device)
            feats_lengths = feats_lengths.to(self.device)

            if self.mode == 'attention':
                hyps = self.model.recognize(
                    feats,
                    feats_lengths,
                    beam_size=self.beam_size,
                    decoding_chunk_size=self.decoding_chunk_size,
                    num_decoding_left_chunks=self.num_decoding_left_chunks,
                    simulate_streaming=self.simulate_streaming)
                hyps = [hyp.tolist() for hyp in hyps]
            elif self.mode == 'ctc_greedy_search':
                hyps = self.model.ctc_greedy_search(
                    feats,
                    feats_lengths,
                    decoding_chunk_size=self.decoding_chunk_size,
                    num_decoding_left_chunks=self.num_decoding_left_chunks,
                    simulate_streaming=self.simulate_streaming)
            # ctc_prefix_beam_search and attention_rescoring only return one
            # result in List[int], change it to List[List[int]] for compatible
            # with other batch decoding mode
            elif self.mode == 'ctc_prefix_beam_search':
                assert (feats.size(0) == 1)
                hyp = self.model.ctc_prefix_beam_search(
                    feats,
                    feats_lengths,
                    beam_size=self.beam_size,
                    decoding_chunk_size=self.decoding_chunk_size,
                    num_decoding_left_chunks=self.num_decoding_left_chunks,
                    simulate_streaming=self.simulate_streaming)
                hyps = [hyp]
            elif self.mode == 'attention_rescoring':
                assert (feats.size(0) == 1)
                hyp, ctc_probs = self.model.attention_rescoring(
                    feats,
                    feats_lengths,
                    beam_size=self.beam_size,
                    decoding_chunk_size=self.decoding_chunk_size,
                    num_decoding_left_chunks=self.num_decoding_left_chunks,
                    ctc_weight=self.ctc_weight,
                    simulate_streaming=self.simulate_streaming)
                hyps = [hyp]
                print("[hyps] len: {}".format(len(hyps)))
                print("[ctc_probs] size: {}".format(ctc_probs.size()))

            for i, key in enumerate(keys):
                result = ""
                for w in hyps[i]:
                    if w == self.eos:
                        break
                    result += self.char_dict[w]

            return result

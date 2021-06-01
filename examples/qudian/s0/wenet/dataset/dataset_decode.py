# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Chao Yang)
# Copyright (c) 2021 Jinsong Pan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import codecs
import copy
import logging
import random

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torchaudio.sox_effects as sox_effects
import yaml
from PIL import Image
from PIL.Image import BICUBIC
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import wenet.dataset.kaldi_io as kaldi_io
from wenet.dataset.wav_distortion import distort_wav_conf
from wenet.utils.common import IGNORE_ID
torchaudio.set_audio_backend("sox")


def _spec_augmentation(x,
                       warp_for_time=False,
                       num_t_mask=2,
                       num_f_mask=2,
                       max_t=50,
                       max_f=10,
                       max_w=80):
    """ Deep copy x and do spec augmentation then return it

    Args:
        x: input feature, T * F 2D
        num_t_mask: number of time mask to apply
        num_f_mask: number of freq mask to apply
        max_t: max width of time mask
        max_f: max width of freq mask
        max_w: max width of time warp

    Returns:
        augmented feature
    """
    y = np.copy(x)
    max_frames = y.shape[0]
    max_freq = y.shape[1]

    # time warp
    if warp_for_time and max_frames > max_w * 2:
        center = random.randrange(max_w, max_frames - max_w)
        warped = random.randrange(center - max_w, center + max_w) + 1

        left = Image.fromarray(x[:center]).resize((max_freq, warped), BICUBIC)
        right = Image.fromarray(x[center:]).resize((max_freq,
                                                   max_frames - warped),
                                                   BICUBIC)
        y = np.concatenate((left, right), 0)
    # time mask
    for i in range(num_t_mask):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        y[start:end, :] = 0
    # freq mask
    for i in range(num_f_mask):
        start = random.randint(0, max_freq - 1)
        length = random.randint(1, max_f)
        end = min(max_freq, start + length)
        y[:, start:end] = 0
    return y

def _spec_substitute(x, max_t=20, num_t_sub=3):
    """ Deep copy x and do spec substitute then return it

    Args:
        x: input feature, T * F 2D
        max_t: max width of time substitute
        num_t_sub: number of time substitute to apply

    Returns:
        augmented feature
    """
    y = np.copy(x)
    max_frames = y.shape[0]
    for i in range(num_t_sub):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        # only substitute the earlier time chosen randomly for current time
        pos = random.randint(0, start)
        y[start:end, :] = y[start - pos:end - pos, :]
    return y

def _waveform_distortion(waveform, distortion_methods_conf):
    """ Apply distortion on waveform

    This distortion will not change the length of the waveform.

    Args:
        waveform: numpy float tensor, (length,)
        distortion_methods_conf: a list of config for ditortion method.
            a method will be randomly selected by 'method_rate' and
            apply on the waveform.

    Returns:
        distorted waveform.
    """
    r = random.uniform(0, 1)
    acc = 0.0
    for distortion_method in distortion_methods_conf:
        method_rate = distortion_method['method_rate']
        acc += method_rate
        if r < acc:
            distortion_type = distortion_method['name']
            distortion_conf = distortion_method['params']
            point_rate = distortion_method['point_rate']
            return distort_wav_conf(waveform, distortion_type,
                                    distortion_conf , point_rate)
    return waveform

# add speed perturb when loading wav
# return augmented, sr
def _load_wav_with_speed(wav_file, speed):
    """ Load the wave from file and apply speed perpturbation

    Args:
        wav_file: input feature, T * F 2D

    Returns:
        augmented feature
    """
    if speed == 1.0:
        return torchaudio.load_wav(wav_file)
    else:
        si, _ = torchaudio.info(wav_file)

        # get torchaudio version
        ta_no = torchaudio.__version__.split(".")
        ta_version = 100 * int(ta_no[0]) + 10 * int(ta_no[1])

        if ta_version < 80:
            # Note: deprecated in torchaudio>=0.8.0
            E = sox_effects.SoxEffectsChain()
            E.append_effect_to_chain('speed', speed)
            E.append_effect_to_chain("rate", si.rate)
            E.set_input_file(wav_file)
            wav, sr = E.sox_build_flow_effects()
        else:
            # Note: enable in torchaudio>=0.8.0
            wav, sr = sox_effects.apply_effects_file(wav_file,
                                                     [['speed', str(speed)],
                                                      ['rate', str(si.rate)]])

        # sox will normalize the waveform, scale to [-32768, 32767]
        wav = wav * (1 << 15)
        return wav, sr


def _extract_feature(batch, wav_distortion_conf,
                     feature_extraction_conf):
    """ Extract acoustic fbank feature from origin waveform.

    Speed perturbation and wave amplitude distortion is optional.

    Args:
        batch: a list of tuple (wav id , wave path).
        speed_perturb: bool, whether or not to use speed pertubation.
        wav_distortion_conf: a dict , the config of wave amplitude distortion.
        feature_extraction_conf:a dict , the config of fbank extraction.

    Returns:
        (keys, feats, labels)
    """
    keys = []
    feats = []
    lengths = []
    wav_dither = wav_distortion_conf['wav_dither']
    wav_distortion_rate = wav_distortion_conf['wav_distortion_rate']
    distortion_methods_conf = wav_distortion_conf['distortion_methods']

    for i, x in enumerate(batch):
        try:
            waveform, sample_rate = torchaudio.load_wav(x[1])
            if wav_distortion_rate > 0.0:
                r = random.uniform(0, 1)
                if r < wav_distortion_rate:
                    waveform = waveform.detach().numpy()
                    waveform = _waveform_distortion(waveform, distortion_methods_conf)
                    waveform = torch.from_numpy(waveform)
            mat = kaldi.fbank(
                waveform,
                num_mel_bins=feature_extraction_conf['mel_bins'],
                frame_length=feature_extraction_conf['frame_length'],
                frame_shift=feature_extraction_conf['frame_shift'],
                dither=wav_dither,
                energy_floor=0.0,
                sample_frequency=sample_rate
            )
            mat = mat.detach().numpy()
            feats.append(mat)
            keys.append(x[0])
            lengths.append(mat.shape[0])
        except (Exception) as e:
            print(e)
            logging.warning('read utterance {} error'.format(x[0]))
            pass

    # Sort it because sorting is required in pack/pad operation
    order = np.argsort(lengths)[::-1]
    sorted_keys = [keys[i] for i in order]
    sorted_feats = [feats[i] for i in order]
    # labels = [x[2].split() for x in batch]
    # labels = [np.fromiter(map(int, x), dtype=np.int32) for x in labels]
    # sorted_labels = [labels[i] for i in order]
    return sorted_keys, sorted_feats


def _load_feature(batch):
    """ Load acoustic feature from files.

    The features have been prepared in previous step, usualy by Kaldi.

    Args:
        batch: a list of tuple (wav id , feature ark path).

    Returns:
        (keys, feats, labels)
    """
    keys = []
    feats = []
    lengths = []
    for i, x in enumerate(batch):
        try:
            mat = kaldi_io.read_mat(x[1])
            feats.append(mat)
            keys.append(x[0])
            lengths.append(mat.shape[0])
        except (Exception):
            # logging.warn('read utterance {} error'.format(x[0]))
            pass
    # Sort it because sorting is required in pack/pad operation
    order = np.argsort(lengths)[::-1]
    sorted_keys = [keys[i] for i in order]
    sorted_feats = [feats[i] for i in order]
    # labels = [x[2].split() for x in batch]
    # labels = [np.fromiter(map(int, x), dtype=np.int32) for x in labels]
    # sorted_labels = [labels[i] for i in order]
    return sorted_keys, sorted_feats


class CollateFunc(object):
    """ Collate function for AudioDataset
    """
    def __init__(self,
                 feature_dither=0.0,
                 speed_perturb=False,
                 spec_aug=False,
                 spec_aug_conf=None,
                 spec_sub=False,
                 spec_sub_conf=None,
                 raw_wav=True,
                 feature_extraction_conf=None,
                 wav_distortion_conf=None,
                 ):
        """
        Args:
            raw_wav:
                    True if input is raw wav and feature extraction is needed.
                    False if input is extracted feature
        """
        self.wav_distortion_conf = wav_distortion_conf
        self.feature_extraction_conf = feature_extraction_conf
        self.spec_aug = spec_aug
        self.feature_dither = feature_dither
        self.speed_perturb = speed_perturb
        self.raw_wav = raw_wav
        self.spec_aug_conf = spec_aug_conf
        self.spec_sub = spec_sub
        self.spec_sub_conf = spec_sub_conf

    def __call__(self, batch):
        assert (len(batch) == 1)
        if self.raw_wav:
            keys, xs = _extract_feature(batch[0],
                                        self.wav_distortion_conf,
                                        self.feature_extraction_conf)
        else:
            keys, xs = _load_feature(batch[0])

        # train_flag = True
        # if ys is None:
        #     train_flag = False

        # # optional feature dither d ~ (-a, a) on fbank feature
        # # a ~ (0, 0.5)
        # if self.feature_dither != 0.0:
        #     a = random.uniform(0, self.feature_dither)
        #     xs = [x + (np.random.random_sample(x.shape) - 0.5) * a for x in xs]
        #
        # # optinoal spec substitute
        # if self.spec_sub:
        #     xs = [_spec_substitute(x, **self.spec_sub_conf) for x in xs]
        #
        # # optinoal spec augmentation
        # if self.spec_aug:
        #     xs = [_spec_augmentation(x, **self.spec_aug_conf) for x in xs]

        # padding
        xs_lengths = torch.from_numpy(
            np.array([x.shape[0] for x in xs], dtype=np.int32))

        # pad_sequence will FAIL in case xs is empty
        if len(xs) > 0:
            xs_pad = pad_sequence([torch.from_numpy(x).float() for x in xs], True, 0)
        else:
            xs_pad = torch.Tensor(xs)

        return keys, xs_pad, xs_lengths


class AudioDataset(Dataset):
    def __init__(self, data_file):
        """
        Dataset for loading audio data.
        """
        self.mini_batch = []

        # Open in utf8 mode since meet encoding problem
        print("[AudioDataset] data_file: {}".format(data_file))

        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                text_list = line.strip().split(" ")
                self.mini_batch.append([])
                self.mini_batch[-1].append((text_list[0], text_list[1]))

        print("[AudioDataset] data len: {}".format(len(self.mini_batch)))

    def __len__(self):
        return len(self.mini_batch)

    def __getitem__(self, idx):
        return self.mini_batch[idx]

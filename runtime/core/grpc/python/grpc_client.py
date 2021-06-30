#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import logging
import argparse
import grpc
import torch
import torchaudio
import time
import numpy as np
from wenet_package import wenet_pb2
from wenet_package import wenet_pb2_grpc
from common import version
from common import constants as const
from utils.log_utils import set_log_filename


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
set_log_filename(logger, "./logs/client.log")


class GrpcClient(object):
    def __init__(self, host: str, port: int, sample_rate: int, n_best: int = 1, continuous_decoding: bool = True):
        self.host = host
        self.port = port
        self.n_best = n_best
        self.continuous_decoding = continuous_decoding
        self.sample_rate = sample_rate
        self.interval = 0.5
        self.ratio = 0.6
        self.sample_interval = int(self.interval * self.sample_rate)

        self.done = False
        self.stub = None
        self.result = []

        self._connect()

    def _connect(self):
        if not self.stub:
            channel = grpc.insecure_channel('{}:{}'.format(self.host, self.port))
            self.stub = wenet_pb2_grpc.ASRStub(channel)
        else:
            logger.error("[_connect] self.stub already init. not init again. ")

    def version(self):
        logger.debug("[version] {}".format(version.version))
        return version.version

    def generate_decode_config(self):
        request = wenet_pb2.Request()
        request.decode_config.nbest_config = self.n_best
        request.decode_config.continuous_decoding_config = self.continuous_decoding
        return request

    def generate_data(self, wav_file):
        logger.debug("[generate_data] wav_file: {}".format(wav_file))

        waveform = torchaudio.load_wav(wav_file)
        wav_data = waveform[0]
        sample_rate = waveform[1]

        if sample_rate == self.sample_rate:
            logger.debug("[generate_data] wav_data: {}".format(wav_data))
            logger.debug("[generate_data] sample_rate: {}".format(sample_rate))

            wav_data_list = torch.split(wav_data, self.sample_interval, dim=1)

            messages = []
            request = self.generate_decode_config()
            messages.append(request)

            for wav_data in wav_data_list:
                # logger.debug("[generate_data] wav_data size: {}".format(wav_data.size()))
                # for i in range(5):
                #     logger.debug("{}    {}".format(i, np.int16(wav_data.numpy())[0, i]))
                request = wenet_pb2.Request()
                data_bytes = bytes(np.int16(wav_data.numpy()))
                request.audio_data = data_bytes
                messages.append(request)

            self.done = False

            for msg in messages:
                # logger.debug("[generate_data] audio_data: {}".format(msg.audio_data))
                logger.debug("[generate_data] type:{} len:{}".format(type(msg.audio_data), len(msg.audio_data)))
                yield msg
                time.sleep(self.interval * self.ratio)
        else:
            logger.error("[generate_data] wav sample_rate: {} != default sample_rate: {}".format(sample_rate,
                                                                                                 self.sample_rate))

    def get_result(self):
        return self.result

    def is_done(self):
        return self.done

    def join(self):
        while True:
            if self.is_done():
                break

    def transform_result(self, response):
        sentence = dict()
        sentence[const.SENTENCE] = response.nbest[0].sentence
        word_pieces = response.nbest[0].wordpieces

        logger.debug("[word_pieces] type: {}; len: {}".format(type(word_pieces), len(word_pieces)))

        start_time = const.MAX_INT
        end_time = const.MIN_INT
        for word_piece in word_pieces:
            # logger.debug("word:{} start:{} end:{}".format(word_piece.word, word_piece.start, word_piece.end))
            start_time = word_piece.start if word_piece.start < start_time else start_time
            end_time = word_piece.end if word_piece.end > end_time else end_time

        sentence[const.START_TIME] = start_time
        sentence[const.END_TIME] = end_time
        self.result.append(sentence)

    def handle_response(self, responses):
        for response in responses:
            logger.debug("[handle_response] status: {}, type: {}".format(response.status, response.type))
            if response.status != const.STATUS_OK:
                logger.error("[handle_response] get response.status: {} != const.STATUS_OK ".format(response.status))
                break

            if response.type == const.TYPE_SERVER_READY:
                self.done = False
                continue
            elif response.type == const.TYPE_PARTIAL_RESULT:
                continue
            elif response.type == const.TYPE_FINAL_RESULT:
                self.transform_result(response)
            elif response.type == const.TYPE_SPEECH_END:
                self.done = True
                break

    def recognize(self, wav_file):
        if self.stub:
            responses = self.stub.Recognize(self.generate_data(wav_file=wav_file))
            self.handle_response(responses=responses)
        else:
            logger.error("[recognize] self.stub is null, use func _connect first. ")
        return


def parse_argvs():
    parser = argparse.ArgumentParser(description='grpc client')
    parser.add_argument("--host", help="host", default="127.0.0.1")
    parser.add_argument("--port", help="port", default="10085")
    parser.add_argument("--sample_rate", help="sample_rate", type=int, default=8000)
    parser.add_argument("--n_best", help="n_best", type=int, default=1)
    parser.add_argument("--continuous_decoding", help="continuous_decoding", type=bool, default=True)
    parser.add_argument("--wav_file", help="wav_file",
                        default="/home/suser/ASR/QdData/test_cyl/6786883706830237706-seat.wav")

    args = parser.parse_args()
    logger.info('[args] {}'.format(args))

    return parser, args


if __name__ == "__main__":
    parser, args = parse_argvs()
    client = GrpcClient(host=args.host,
                        port=args.port,
                        sample_rate=args.sample_rate,
                        n_best=args.n_best,
                        continuous_decoding=args.continuous_decoding)
    time_1 = int(time.time() * 1000)
    client.recognize(wav_file=args.wav_file)
    time_2 = int(time.time() * 1000)
    client.join()
    time_3 = int(time.time() * 1000)
    logger.info(client.get_result())
    logger.info("[time use] total: {}: recognize: {}, join: {}".format(time_3-time_1, time_2-time_1, time_3-time_2))


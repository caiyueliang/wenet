# -*- coding: utf-8 -*-
""""
-------------------------------------------------
   Description :  wenet client
   Author :       caiyueliang
   Date :         2021-06-17
-------------------------------------------------
"""
from __future__ import print_function
import sys
import argparse
import logging
import time

import grpc
from wenet_package.wenet_pb2 import Request, Response
from wenet_package.wenet_pb2_grpc import WenetServiceStub
from utils.log_utils import set_log_filename

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
set_log_filename(logger, "./logs/client.log")


def request(ip, port, oss):
    start_time = time.time()
    logger.debug("[request] {}:{} start ... ".format(ip, port))
    # 使用with语法保证channel自动close
    with grpc.insecure_channel(ip + ':' + port) as channel:
        # 客户端通过stub来实现rpc通信
        stub = WenetServiceStub(channel)

        # 客户端必须使用定义好的类型，这里是HelloRequest类型
        response = stub.Recognize(Request(oss=oss))

    end_time = time.time()
    time_used = end_time - start_time
    logger.debug("[request][time: {}s] client received: {}".format(time_used, response.message))
    print("[request][time: {}s] client received: {}".format(time_used, response.message))


def parse_argvs():
    parser = argparse.ArgumentParser(description='wenet server')
    parser.add_argument('--ip', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=str, default='50001')
    parser.add_argument('--oss', type=str, default='oss://bi-ai-data/caiyueliang/QdData/wav_1/6686505489037045768-seat.wav')
    logger.info("[parse_argvs] {}".format(' '.join(sys.argv)))
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_argvs()
    request(ip=args.ip, port=args.port, oss=args.oss)

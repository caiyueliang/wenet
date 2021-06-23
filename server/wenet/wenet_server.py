# -*-coding: utf-8 -*-
""""
-------------------------------------------------
   Description :  wenet server
   Author :       caiyueliang
   Date :         2021-06-17
-------------------------------------------------
"""
import os
import sys
import argparse
from concurrent import futures
import grpc
import logging
import time
from wenet_package.wenet_pb2_grpc import add_WenetServiceServicer_to_server, WenetServiceServicer
from wenet_package.wenet_pb2 import Request, Response
from manager.asr_manager import ASRManager
from utils.log_utils import set_log_filename
from utils.oss_utils import download

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
set_log_filename(logger, "./logs/server.log")


class Hello(WenetServiceServicer):
    def __init__(self):
        self.tmp_dir = "tmp"
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.asr_manager = ASRManager(config="./conf/server.yaml")

    # 这里实现我们定义的接口
    def Recognize(self, request, context):
        try:
            oss = request.oss
            logger.debug("[Recognize] oss: {}".format(oss))
            local_path = os.path.join(self.tmp_dir, oss.split("/")[-1])
            success = download(oss=oss, local_path=local_path)
            if success:
                logger.debug("[Recognize] download oss success ... ")
                result = self.asr_manager.recognize(wav_path=local_path)
            else:
                logger.error("[Recognize] download oss failed ... ")
                result = "[error] download from oss failed ."
        except Exception as ex:
            logger.error("[exception] see log for details. ")
            logger.exception(ex)
            result = "[exception] see log for details. "

        return Response(message=result)


def serve(ip="0.0.0.0", port="5000", max_workers=10):
    # 这里通过thread pool来并发处理server的任务
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    # 将对应的任务处理函数添加到rpc server中
    add_WenetServiceServicer_to_server(servicer=Hello(), server=server)

    # 这里使用的非安全接口，世界gRPC支持TLS/SSL安全连接，以及各种鉴权机制
    server.add_insecure_port(ip + ':' + port)
    server.start()
    logger.info("[serve] {}:{} start ... ".format(ip, port))
    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        server.stop(0)


def parse_argvs():
    parser = argparse.ArgumentParser(description='wenet server')
    parser.add_argument('--ip', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=str, default='50001')
    parser.add_argument('--max_workers', type=int, default=10)
    logger.info("[parse_argvs] {}".format(' '.join(sys.argv)))
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_argvs()

    serve(ip=args.ip, port=args.port, max_workers=args.max_workers)

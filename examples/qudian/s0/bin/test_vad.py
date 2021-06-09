# -*- coding: utf-8 -*-
""""
-------------------------------------------------
   Description :  批量预测vad
   Author :       caiyueliang
   Date :         2021-06-09
-------------------------------------------------
"""
import os
import logging
import argparse
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class VADTester(object):
    def __init__(self, exe_path, host, port, wav_dir, out_dir):
        self.wav_dir = wav_dir
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.tmp_dir = "./tmp/"
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.exe_path = exe_path
        self.host = host
        self.port = port
        self.cmd = '{} --host {} --port {} --continuous_decoding "true" --wav_path "{}" > {} 2>&1 '
        return

    def _get_all_file_list(self, key=None):
        file_list = list()

        for dir_path, dir_names, file_names in os.walk(self.wav_dir):
            for file_name in file_names:
                if key:
                    if key in file_name:
                        file_path = os.path.join(dir_path, file_name)
                        file_list.append(file_path)
                else:
                    file_path = os.path.join(dir_path, file_name)
                    file_list.append(file_path)

        return file_list

    def recognize(self, file, tmp_file):
        exe_cmd = self.cmd.format(self.exe_path, self.host, self.port, file, tmp_file)
        logging.warning("[exe_cmd] {}".format(exe_cmd))
        os.system(exe_cmd)

    def transform_to_json(self, tmp_file, json_file):
        result_list = list()

        with open(tmp_file, mode="r", encoding="utf-8") as fr:
            text_list = fr.readlines()

            for text in text_list:
                text = text.strip().replace("[final_result] ", "").strip("\"").replace("\\", "")
                # print(text)
                text_json_list = json.loads(text)
                if len(text_json_list) > 0:
                    text_json = text_json_list[0]
                    # print(text_json)

                    if 'word_pieces' in text_json.keys() and len(text_json['sentence']) > 0:
                        start_time = 9999000
                        end_time = 0
                        for word in text_json['word_pieces']:
                            if start_time > word['start']:
                                start_time = word['start']
                            if end_time < word['end']:
                                end_time = word['end']
                        text_json['start'] = start_time
                        text_json['end'] = end_time
                        del text_json['word_pieces']

                        result_list.append(text_json)

        with open(json_file, mode="w", encoding="utf-8") as fw:
            json.dump(result_list, fw, ensure_ascii=False, indent=4)

        return

    def workflow(self):
        file_list = self._get_all_file_list(key=".wav")

        for file in file_list:
            logging.warning("[recognize] file: {}".format(file))
            key = str(file.split(os.sep)[-1]).split(".")[0]
            tmp_file = os.path.join(self.tmp_dir, key + ".txt")
            json_file = os.path.join(self.out_dir, key + ".json")

            # self.recognize(file=file, tmp_file=tmp_file)
            self.transform_to_json(tmp_file, json_file)


def parse_argvs():
    parser = argparse.ArgumentParser(description='test vad')
    parser.add_argument("--exe_path", help="exe path", type=str,
                        default="/DATA/caiyueliang/wenet/runtime/server/x86/build/websocket_client_main")
    parser.add_argument("--host", help="host", type=str, default='127.0.0.1')
    parser.add_argument("--port", help="port", type=str, default='10086')
    parser.add_argument("--wav_dir", help="wav dir", type=str, default='/DATA/ASR/QdData/test_cyl/')
    parser.add_argument("--out_dir", help="out dir", type=str, default='./result')
    args = parser.parse_args()
    logging.warning(args)
    return parser, args


if __name__ == "__main__":
    parser, args = parse_argvs()

    vad_tester = VADTester(exe_path=args.exe_path, host=args.host, port=args.port, wav_dir=args.wav_dir,
                           out_dir=args.out_dir)

    vad_tester.workflow()

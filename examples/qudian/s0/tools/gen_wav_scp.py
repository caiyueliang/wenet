# -*- coding: utf-8 -*-
""""
-------------------------------------------------
   Description :  gen wav scp
   Author :       caiyueliang
   Date :         2021-05-29
-------------------------------------------------
"""

import os
import argparse
import logging
import numpy as np

logging.basicConfig(level=logging.NOTSET)


class WavScpGenerator(object):
    def __init__(self, wav_dir, out_dir, split):
        self.wav_dir = wav_dir
        self.out_dir = out_dir
        self.split = split

        os.makedirs(self.out_dir, exist_ok=True)

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

    def _split_n(self, file_list, split):
        file_groups = []
        file_len = len(file_list)
        num = file_len % split
        append_list = [1 if x < num else 0 for x in range(split)]
        n = file_len // split
        for i in range(split):
            sub_file_list = file_list[i*n + sum(append_list[0:i]): (i+1)*n + sum(append_list[0:i+1])]
            file_groups.append(sub_file_list)
        return file_groups

    def write_wav_scp(self, output_file, files):
        with open(output_file, mode="w", encoding="utf-8") as fw:
            for file in files:
                key = file.split(os.sep)[-1].split(".")[0]
                fw.write(key + " " + file + "\n")
        return

    def workflow(self):
        file_list = self._get_all_file_list(key=".wav")

        if self.split == 1:
            output_file = os.path.join(self.out_dir, "wav.0.scp")
            self.write_wav_scp(output_file=output_file, files=file_list)
            print("[split:{}] len: {}".format(1, len(file_list)))
        else:
            file_groups = self._split_n(file_list=file_list, split=self.split)

            count = 0
            for i in range(len(file_groups)):
                sub_len = len(file_groups[i])
                count += count
                print("[split][{}/{}] sub_len: {}".format(i, len(file_groups), sub_len))
                output_file = os.path.join(self.out_dir, "wav." + str(i) + ".scp")
                self.write_wav_scp(output_file=output_file, files=file_groups[i])
            print("[split:{}] len: {}".format(len(file_groups), count))


def parse_argvs():
    """ 取数时长限制参数 """
    parser = argparse.ArgumentParser(description='gen wav scp')
    parser.add_argument("--wav_dir", help="wav dir", default="/data/ASR/QdData/test/")
    parser.add_argument("--out_dir", help="out_dir", default="./data")
    parser.add_argument("--split", help="split n ", type=int, default=4)
    args = parser.parse_args()
    logging.info('[args] {}'.format(args))

    return parser, args


if __name__ == "__main__":
    parser, args = parse_argvs()
    generator = WavScpGenerator(wav_dir=args.wav_dir, out_dir=args.out_dir, split=args.split)

    generator.workflow()

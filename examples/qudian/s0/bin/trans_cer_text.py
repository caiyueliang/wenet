# -*- coding: utf-8 -*-
""""
-------------------------------------------------
   Description :  趣店格式的数据转换成aishell的格式
   Author :       caiyueliang
   Date :         2021-04-07
-------------------------------------------------
"""
import os
import logging
import argparse


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_argvs():
    parser = argparse.ArgumentParser(description='ASR data prepare')
    parser.add_argument("--input_file", help="input dir", type=str,
                        default='/DATA/disk1/caiyueliang/wenet/examples/qudian/s0/exp/conformer/test_attention/text_8')

    args = parser.parse_args()
    logging.warning(args)
    return parser, args


if __name__ == "__main__":
    parser, args = parse_argvs()

    new_file = args.input_file + "_trans"

    with open(args.input_file, mode="r", encoding="utf-8") as fr:
        with open(new_file, mode="w", encoding="utf-8") as fw:
            lines = fr.readlines()
            for line in lines:
                text_list = line.strip().split(" ")

                if len(text_list) > 1:
                    key = text_list[0]
                    text = text_list[1]
                    new_text = " ".join([i for i in text])
                    new_text = new_text.replace("< u n k >", "<UNK>")

                    fw.write(key + " " + new_text + "\n")
                else:
                    print(line)
                    fw.write(line)

# -*- coding: utf-8 -*-
""""
-------------------------------------------------
   Description :  merge dict
   Author :       caiyueliang
   Date :         2021-04-15
-------------------------------------------------
"""
import os
import logging
import argparse


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def to_dict(dict_list):
    new_dict = dict()
    for line in dict_list:
        line_list = line.strip().split(" ")
        if line_list[0] != "<sos/eos>":
            new_dict[line_list[0]] = int(line_list[1])
    return new_dict


def merge(src_dict, append_dict):
    for key in append_dict.keys():
        if key not in src_dict.keys():
            src_dict[key] = len(src_dict)

    src_dict["<sos/eos>"] = len(src_dict)
    return src_dict


def save_dict(output_file, data_dict):
    dir_name = os.path.dirname(output_file)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(output_file, mode="w", encoding="utf-8") as fw:
        for key in data_dict.keys():
            fw.write("{} {}\n".format(key, data_dict[key]))


def parse_argvs():
    parser = argparse.ArgumentParser(description='ASR data prepare')
    parser.add_argument("--src_dict", help="src_dict path", type=str, default='/DATA/disk1/caiyueliang/wenet/models/20210204_conformer_exp/words.txt')
    parser.add_argument("--append_dict", help="append_dict path", type=str, default='/DATA/disk1/caiyueliang/wenet/examples/qudian/s1_cyl/data/dict/lang_char_3857.txt')
    parser.add_argument("--output_dict", help="output_dict path", type=str, default='./new_words.txt')
    args = parser.parse_args()
    logging.warning(args)
    return parser, args


if __name__ == "__main__":
    parser, args = parse_argvs()

    with open(args.src_dict, mode="r", encoding="utf-8") as fr:
        src_list = fr.readlines()

    with open(args.append_dict, mode="r", encoding="utf-8") as fr:
        append_list = fr.readlines()

    src_dict = to_dict(src_list)
    append_dict = to_dict(append_list)
    print("[src_dict] len: {}".format(len(src_dict)))
    print("[append_dict] len: {}".format(len(append_dict)))

    new_dict = merge(src_dict, append_dict)
    print("[new_dict] len: {}".format(len(new_dict)))
    save_dict(output_file=args.output_dict, data_dict=new_dict)

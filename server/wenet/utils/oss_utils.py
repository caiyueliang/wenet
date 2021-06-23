# -*- coding: utf-8 -*-
import os
import logging

logger = logging.getLogger()


def download(oss, local_path):
    oss_cmd = "~/ossutil64 cp {} {}"
    exec_cmd = oss_cmd.format(oss, local_path)
    logger.debug("[download] exec_cmd: {}".format(exec_cmd))
    os.system(exec_cmd)
    return True

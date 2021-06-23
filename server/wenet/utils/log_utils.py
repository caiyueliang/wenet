import logging
import time
import os
import fcntl

from logging.handlers import TimedRotatingFileHandler


class MultiCompatibleTimedRotatingFileHandler(TimedRotatingFileHandler):
    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        # get the time that this sequence started at and make it a TimeTuple
        currentTime = int(time.time())
        dstNow = time.localtime(currentTime)[-1]
        t = self.rolloverAt - self.interval
        if self.utc:
            timeTuple = time.gmtime(t)
        else:
            timeTuple = time.localtime(t)
            dstThen = timeTuple[-1]
            if dstNow != dstThen:
                if dstNow:
                    addend = 3600
                else:
                    addend = -3600
                timeTuple = time.localtime(t + addend)
        dfn = self.baseFilename + "." + time.strftime(self.suffix, timeTuple)
        # 兼容多进程并发 LOG_ROTATE
        if not os.path.exists(dfn):
            f = open(self.baseFilename, 'a')
            fcntl.lockf(f.fileno(), fcntl.LOCK_EX)
            if not os.path.exists(dfn):
                os.rename(self.baseFilename, dfn)  # 释放锁 释放老log句柄
        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)
        if not self.delay:
            self.stream = self._open()
        newRolloverAt = self.computeRollover(currentTime)
        while newRolloverAt <= currentTime:
            newRolloverAt = newRolloverAt + self.interval
        # If DST changes and midnight or weekly rollover, adjust for this.
        if (self.when == 'MIDNIGHT' or self.when.startswith('W')) and not self.utc:
            dstAtRollover = time.localtime(newRolloverAt)[-1]
            if dstNow != dstAtRollover:
                if not dstNow:  # DST kicks in before next rollover, so we need to deduct an hour
                    addend = -3600
                else:  # DST bows out before next rollover, so we need to add an hour
                    addend = 3600
                newRolloverAt += addend
        self.rolloverAt = newRolloverAt


def set_log_filename(my_logger, filename, when='d', interval=1, backup_count=7) -> None:
    """set log filename """
    log_dir = os.path.dirname(filename)
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s %(pathname)s:%(lineno)d")
    log_file_handler = MultiCompatibleTimedRotatingFileHandler(
        filename=filename, when=when, interval=interval, backupCount=backup_count)
    log_file_handler.setFormatter(formatter)
    log_file_handler.setLevel(logging.DEBUG)
    my_logger.addHandler(log_file_handler)

import logging
from logging import handlers
import time
import datetime
import pytz
import sys
import os

def logger_1(name, level, filename, sh_level, fh_level):
    """
    :param name:  日志收集器名字
    :param level: 日志收集器的等级
    :param filename:  日志文件的名称
    :param sh_level:  控制台输出日志的等级
    :param fh_level:    文件输出日志的等级
    :return: 返回创建好的日志收集器
    """

    # 1、创建日志收集器
    log = logging.getLogger(name)

    # 2、创建日志收集器的等级
    log.setLevel(level=level)

    # 3、创建日志收集渠道和等级
    sh = logging.StreamHandler()
    sh.setLevel(level=sh_level)
    log.addHandler(sh)
    fh = logging.FileHandler(filename=filename, encoding="utf-8")
    # fh1 = handlers.TimedRotatingFileHandler(filename=filename,when="D",interval=1,backupCount=10,encoding="utf-8")
    fh.setLevel(level=fh_level)
    log.addHandler(fh)

    # 4、设置日志的输出格式
    run_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S")
    formats = "{} - [%(funcName)s-->line:%(lineno)d] - %(levelname)s:%(message)s".format(run_time)
    log_format = logging.Formatter(fmt=formats)
    sh.setFormatter(log_format)
    fh.setFormatter(log_format)
    return log

class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def logger_2(log_path, file_name):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = log_path + '/' + file_name
    sys.stdout = Logger(log_file)
    sys.stderr = Logger(log_file)

if __name__ == '__main__':
    a = 1
    # log = create_log(name="las_log", level=logging.DEBUG, filename="../log/data_log/log1.txt", sh_level=logging.DEBUG,
    #                  fh_level=logging.DEBUG)
    # log.info(msg="--------debug--------")
    # log.info(msg="--------info--------")
    # log.info(msg="--------warning--------")
    # log.info(msg="--------error--------")
    # log.info(msg="--------critical--------")
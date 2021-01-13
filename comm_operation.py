#!/usr/bin/python3
# -*- coding:utf-8 -*-
import sys
import tensorflow as tf


def set_device(gpu_id=0, gpu_mem=None, only_cpu=False, growth=False):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    cpus = tf.config.experimental.list_physical_devices('CPU')
    if not only_cpu:
        tf.config.experimental.set_visible_devices(devices=gpus[gpu_id], device_type='GPU')
        if not growth:
            tf.config.experimental.set_virtual_device_configuration(gpus[gpu_id], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_mem)])
        else:
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
    else:
        tf.config.experimental.set_visible_devices(devices=cpus, device_type='CPU')
        tf.config.experimental.set_visible_devices(devices=[], device_type='GPU')


def input_bool(s: str):
    if s.lower() == "true" or s == "1" or s.lower() == 'y' or s.lower() == 'yes':
        res = True
    elif s.lower() == "false" or s == "0" or s.lower() == 'n' or s.lower() == 'yes':
        res = False
    else:
        raise ValueError("无效输入")
    return res


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a+')  # 不重写文件，以添加的形式加入文件末尾  'w'则重写文件

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def redirect2log(filename):
    """
    将控制台输出保存到文件 filename中
    """
    sys.stdout = Logger(filename, sys.stdout)

    return

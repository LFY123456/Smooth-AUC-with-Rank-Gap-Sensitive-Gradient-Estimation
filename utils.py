
import os
import argparse
import pickle
from collections import deque
import random
from time import time
from datetime import datetime
import numpy as np
import torch.nn as nn
import torch
from torch import optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
# from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
import logging

def logCof(logger, log_path="./log/", log_file_name="test.log"):
    '''
        CRITICAL, ERROR, WARNING, INFO, DEBUG
        从左到右，日志级别一次下降
        打印的时候只会打印比设置的级别高的日志
        logger设置的级别是最低级别，它的handler比他低也没用，不会输出更低的

    '''
    logger.setLevel(level=logging.DEBUG)

    if log_file_name == "test.log":
        log_file_name = "test_" + datetime.now().strftime('%Y%m%d%H%M%S') + ".log"
    os.makedirs(log_path, exist_ok=True)
    handler = logging.FileHandler(os.path.join(log_path, log_file_name))
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # console = logging.StreamHandler()
    # console.setLevel(logging.WARNING)
    # console.setFormatter(formatter)

    logger.addHandler(handler)
    # logger.addHandler(console)
    return logger


class Auc(object):
    """
    分段，计算总的auc；当计算特别大的测试集auc时可以使用；来源paddle。


    下面是测试样例
    label1 = np.random.randint(low=0, high=2, size=[1000])
    predict1 = np.random.uniform(0, 1, size=[1000])
    label2 = np.random.randint(low=0, high=2, size=[1000])
    predict2 = np.random.uniform(0, 1, size=[1000])
    label = np.hstack((label1, label2))
    predict = np.hstack((predict1, predict2))
    auc = Auc(num_buckets=102400)
    t = time()
    auc.Update(label1, predict1)
    print(auc.Compute())
    print("this cost", time() - t, "s")
    t = time()
    auc.Update(label2, predict2)
    print(auc.Compute())
    print("this cost", time() - t, "s")
    t = time()
    print(sklearn_metrics.roc_auc_score(label, predict))
    print("sklearn auc cost", time() - t, "s")
    """

    def __init__(self, num_buckets):
        self._num_buckets = num_buckets
        self._table = np.zeros(shape=[2, self._num_buckets])

    def Reset(self):
        self._table = np.zeros(shape=[2, self._num_buckets])

    def Update(self, labels: np.ndarray, predicts: np.ndarray):
        """
        :param labels: 1-D ndarray
        :param predicts: 1-D ndarray
        :return: None
        """
        labels = labels.astype(np.int)
        predicts = self._num_buckets * predicts

        buckets = np.round(predicts).astype(np.int)
        buckets = np.where(buckets < self._num_buckets,
                           buckets, self._num_buckets - 1)

        for i in range(len(labels)):
            self._table[labels[i], buckets[i]] += 1

    def Compute(self):
        tn = 0
        tp = 0
        area = 0
        for i in range(self._num_buckets):
            new_tn = tn + self._table[0, i]
            new_tp = tp + self._table[1, i]
            # self._table[1, i] * tn + self._table[1, i]*self._table[0, i] / 2
            area += (new_tp - tp) * (tn + new_tn) / 2
            tn = new_tn
            tp = new_tp
        if tp < 1e-3 or tn < 1e-3:
            return -0.5  # 样本全正例，或全负例
        return area / (tn * tp)


def clear_result(k=5):
    import os
    import pandas as pd
    '''
        每个实验保留最好的前k个结果，其余删除。
    '''
    list = os.listdir("./saved_models")
    list = [x.split("_") for x in list]
    df = pd.DataFrame(list, columns=["dataset", "model", "loss", "time", "perform"])
    ans = []
    for name, data in df.groupby(["dataset", "loss"]):
        data.perform = data.perform.apply(lambda x: float(x[:-3]))
        temp = data.sort_values("perform", ascending=False)[k:]
        ans.extend(temp.index)
    temp = df.iloc[ans, :]
    ans = temp["dataset"] + "_" + temp["model"] + "_" + temp["loss"] + "_" + temp["time"] + "_" + temp["perform"]
    ans = ["./saved_models/" + x for x in ans]
    for file in ans:
        os.remove(file)

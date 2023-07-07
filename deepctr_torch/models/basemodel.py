# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,weichenswc@163.com

"""
from __future__ import print_function

import pandas as pd
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import nni

try:
    from tensorflow.python.keras.callbacks import CallbackList
except ImportError:
    from tensorflow.python.keras._impl.keras.callbacks import CallbackList

from ..inputs import build_input_features, SparseFeat, DenseFeat, VarLenSparseFeat, get_varlen_pooling_list, \
    create_embedding_matrix, varlen_embedding_lookup
from ..layers import PredictionLayer
from ..layers.utils import slice_arrays
from ..callbacks import History


class SmoothAUCLoss(nn.Module):

    def __init__(self):
        super(SmoothAUCLoss, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, y_pred, y_true, tau=0.02, reduction="sum"):
        assert y_pred.shape == y_true.shape, "smooth auc loss 输入的两个label集合大小不等！！！"
        pos_set = torch.masked_select(y_pred, y_true.bool())
        neg_set = torch.masked_select(y_pred, ~y_true.bool())
        res_matrix = (pos_set.view(pos_set.shape[0], 1) - neg_set.view(1, neg_set.shape[0])).to(torch.float64)
        return 1 - torch.div(torch.sum(self.sigmoid(res_matrix/tau)), pos_set.shape[0] * neg_set.shape[0])

class SmoothAUCLossLambda(nn.Module):

    def __init__(self):
        super(SmoothAUCLossLambda, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, sui, suj, tau=0.02):
        '''
        单个用户的 weighted_sauc loss 计算
        sui: torch.tensor(pos_num, 1)
        suj: torch.tensor(1, neg_num)
        '''
        sui = sui.reshape(-1, 1)
        suj = suj.reshape(1, -1)
        assert len(sui.shape) == 2 and sui.shape[1] == 1, f"sui.shape=={sui.shape}"
        assert len(suj.shape) == 2 and suj.shape[0] == 1, f"sui.shape=={suj.shape}"

        residual = sui - suj
        pos_neg_mat = self.sigmoid(residual / tau)

        lambda_weight = self.posrank(sui, suj).detach()

        return 1 - torch.mean(torch.mul(pos_neg_mat, lambda_weight)), - torch.sum(torch.mul(pos_neg_mat, lambda_weight)), 1 - torch.sum(pos_neg_mat) / (sui.shape[0] * suj.shape[1])
        # return 1 - torch.sum(torch.mul(pos_neg_mat, lambda_weight))

    def posrank(self, sui, suj):
        sui = sui.reshape(-1)
        suj = suj.reshape(-1)
        data = torch.cat([sui, suj])
        _, idx = data.sort(descending=False)
        _, rank = idx.sort()  # 再对索引升序排列，得到其索引作为排名rank
        k = len(sui)
        return torch.abs(rank[:k].reshape(-1, 1) - rank[k:].reshape(1, -1)) / (len(sui) * len(suj))


class BPR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_ui, x_uj):
        x_uij = x_ui - x_uj
        log_prob = F.logsigmoid(x_uij).mean()
        return -log_prob

class Linear(nn.Module):
    def __init__(self, feature_columns, feature_index, init_std=0.0001, device='cpu'):
        super(Linear, self).__init__()
        self.feature_index = feature_index
        self.device = device
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False,
                                                      device=device)

        #         nn.ModuleDict(
        #             {feat.embedding_name: nn.Embedding(feat.dimension, 1, sparse=True) for feat in
        #              self.sparse_feature_columns}
        #         )
        # .to("cuda:1")
        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), 1).to(
                device))
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, X, sparse_feat_refine_weight=None):

        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()
        ) for feat in self.sparse_feature_columns]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.varlen_sparse_feature_columns)
        varlen_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                        self.varlen_sparse_feature_columns, self.device)

        sparse_embedding_list += varlen_embedding_list

        linear_logit = torch.zeros([X.shape[0], 1]).to(sparse_embedding_list[0].device)
        if len(sparse_embedding_list) > 0:
            sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)
            if sparse_feat_refine_weight is not None:
                # w_{x,i}=m_{x,i} * w_i (in IFM and DIFM)
                sparse_embedding_cat = sparse_embedding_cat * sparse_feat_refine_weight.unsqueeze(1)
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
            linear_logit += sparse_feat_logit
        if len(dense_value_list) > 0:
            dense_value_logit = torch.cat(
                dense_value_list, dim=-1).matmul(self.weight)
            linear_logit += dense_value_logit

        return linear_logit


class BaseModel(nn.Module):
    def __init__(self, linear_feature_columns, dnn_feature_columns, lr=0.01, l2_reg_linear=1e-5, l2_reg_embedding=1e-5,
                 init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None):

        super(BaseModel, self).__init__()
        torch.manual_seed(seed)
        self.dnn_feature_columns = dnn_feature_columns

        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.lr = lr
        self.device = device
        self.gpus = gpus
        self.sal = SmoothAUCLoss()
        self.sall = SmoothAUCLossLambda()
        self.bpr = BPR()
        if gpus and str(self.gpus[0]) not in self.device:
            raise ValueError(
                "`gpus[0]` should be the same gpu with `device`")

        self.feature_index = build_input_features(linear_feature_columns + dnn_feature_columns)
        # 下面只为深度模块要用到的特征列，进行。。。
        self.embedding_dict = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False, device=device)
        #         nn.ModuleDict(
        #             {feat.embedding_name: nn.Embedding(feat.dimension, embedding_size, sparse=True) for feat in
        #              self.dnn_feature_columns}
        #         )

        self.linear_model = Linear(linear_feature_columns, self.feature_index, device=device)

        self.regularization_weight = []

        # 为数据加上正则
        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)

        self.out = PredictionLayer(task, )
        self.to(device)

        # parameters for callbacks
        self._is_graph_network = True  # used for ModelCheckpoint in tf2
        self._ckpt_saved_epoch = False  # used for EarlyStopping in tf1.14
        self.history = History()

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None):
        """

        :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
        :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
        :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
        :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`

        :return: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))

        else:
            val_x = []
            val_y = []
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            # myloss_set = []
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()

                        y_pred = model(x).squeeze()

                        optim.zero_grad()
                        # start_time = time.time()
                        loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                        # print(f"counting once of smooth auc loss costs {time.time() - start_time} s.")
                        reg_loss = self.get_regularization_loss()

                        total_loss = loss + reg_loss + self.aux_loss

                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        # myloss_set.append(total_loss.item())
                        total_loss.backward()
                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name == "auc_personal":
                                    continue
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(metric_fun(
                                    y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()
            # print("one epoch loss set: ", myloss_set)
            # print("one epoch auc set: ", train_result["auc"])
            # Add epoch_logs
            if not isinstance(self.loss_func, str):
                epoch_logs["loss"] = total_loss_epoch / sample_num
            else:
                epoch_logs["loss"] = total_loss_epoch / steps_per_epoch

            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                # eval_result = self.evaluate(val_x, val_y, batch_size)
                eval_result = self.evaluate_personal(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    if name == "auc_personal":
                        continue
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name + \
                                    ": {0: .4f}".format(epoch_logs["val_" + name])
                print(eval_str)
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()

        return self.history

    def fit_BCE(self, logger, train_data_pos=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None,lr=0.01):
        """

        :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
        :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
        :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
        :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`

        :return: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """

        train_data_pos: pd.DataFrame  # (sample_num x (uid, iid, label=1))

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]
        else:
            val_x = []
            val_y = []

        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        # optim = self.optim
        optim = torch.optim.Adam(self.parameters(), lr=lr)

        if self.gpus:
            logger.warning('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            logger.warning(self.device)

        # 重排成  users_num x [interact_items]
        train_data_pos.reset_index(drop=True, inplace=True)
        interacn_matrix = train_data_pos.groupby(["userInt"]).newsInt.unique()
        train_tensor_pos = torch.from_numpy(np.array(train_data_pos))

        sample_num = train_data_pos.shape[0]*2    # 1:1 采样，所以样本数是正样本的两倍
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        # Train
        logger.warning("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            sample_num, 0, steps_per_epoch))
        best_val_score = 0
        best_model_params = None

        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}

            neg_data = pd.DataFrame(train_data_pos, copy=True)
            neg_data.label = 0
            idx = 0
            for _ in range(len(train_data_pos)):
                while True:
                    neg_idx = random.randint(0, 16980)
                    if neg_idx not in interacn_matrix[train_data_pos.loc[idx, "userInt"]]:
                        neg_data.loc[idx, "newsInt"] = neg_idx
                        idx += 1
                        break
            data = pd.concat([train_data_pos, neg_data], axis=0)

            train_tensor_data = Data.TensorDataset(torch.from_numpy(np.array(data.drop(columns=["label"]))),
                                                   torch.from_numpy(np.array(data[["label"]])))
            train_loader = DataLoader(dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:
                        # assert sum(y_train > 1) == 0
                        # assert sum(y_train < 0) == 0
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()
                        # print(y)
                        y_pred = model(x).squeeze()
                        # assert sum(y_pred >= 1) == 0
                        # assert sum(y_pred <= 0) == 0
                        optim.zero_grad()
                        # start_time = time.time()
                        # print(y_pred)
                        loss = loss_func(y_pred, y.squeeze(), reduction='mean')
                        reg_loss = self.get_regularization_loss()

                        total_loss = loss + reg_loss + self.aux_loss

                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        # nni.report_intermediate_result(total_loss.item())
                        total_loss.backward()
                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name == "auc_personal" or "binary_crossentropy":
                                    continue
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(metric_fun(
                                    y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()
            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / steps_per_epoch

            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                # eval_result = self.evaluate(val_x, val_y, batch_size)
                eval_result = self.evaluate_personal(val_x, val_y)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                logger.warning('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    if name == "auc_personal" or "binary_crossentropy":
                        continue
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name + \
                                    ": {0: .4f}".format(epoch_logs["val_" + name])
                logger.warning(eval_str)
            nni.report_intermediate_result(epoch_logs["val_auc_personal"])
            if epoch_logs["val_auc_personal"] >= best_val_score:
                best_val_score = epoch_logs["val_auc_personal"]
                best_model_params = model.state_dict()
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()

        return self.history, best_val_score, best_model_params

    def fit_BCE_new(self, logger, train_data_pos=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0,
                validation_split=0.,
                validation_data=None, shuffle=True, callbacks=None, item_size=16980,lr=0.01):
        """

        :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
        :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
        :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
        :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`

        :return: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """

        train_data_pos: pd.DataFrame  # (sample_num x (uid, iid, label=1))

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]
        else:
            val_x = []
            val_y = []

        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        # optim = self.optim
        optim = torch.optim.Adam(self.parameters(), lr=lr)

        if self.gpus:
            logger.warning('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            logger.warning(self.device)

        # 重排成  users_num x [interact_items]
        train_data_pos.reset_index(drop=True, inplace=True)
        interacn_matrix = train_data_pos.groupby(["userInt"]).newsInt.unique()
        """different"""
        ui_matrix = {uid: set(interacn_matrix[uid]) for uid in train_data_pos.userInt.unique()}
        print("构造数据集结束")
        # train_tensor_pos = torch.from_numpy(np.array(train_data_pos))

        sample_num = train_data_pos.shape[0] * 2  # 1:1 采样，所以样本数是正样本的两倍
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        # Train
        logger.warning("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            sample_num, 0, steps_per_epoch))
        best_val_score = 0
        best_model_params = None

        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            '''
            x_pos = xxx
            sample
            x_neg = (x, y)
            concate(x_pos, x_neg)
            shuffle
            '''
            neg_data = pd.DataFrame(train_data_pos, copy=True)
            neg_data.label = 0
            idx = 0
            for _ in range(len(train_data_pos)):
                uid = neg_data.iloc[idx, 0]
                while True:
                    neg_idx = random.randint(0, item_size - 1) #原来是16980
                    if neg_idx not in ui_matrix[uid]:
                        neg_data.iloc[idx, 1] = neg_idx
                        idx += 1
                        break
            # neg_data.newsInt = pd.DataFrame(np.random.randint(0, item_size, size=(train_data_pos.shape[0], 1)))
            data = pd.concat([train_data_pos, neg_data], axis=0)
            print("采样结束")
            train_tensor_data = Data.TensorDataset(torch.from_numpy(np.array(data.drop(columns=["label"]))),
                                                   torch.from_numpy(np.array(data[["label"]])))
            train_loader = DataLoader(dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:
                        # assert sum(y_train > 1) == 0
                        # assert sum(y_train < 0) == 0
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()
                        # print(y)
                        y_pred = model(x).squeeze()
                        # assert sum(y_pred >= 1) == 0
                        # assert sum(y_pred <= 0) == 0
                        optim.zero_grad()
                        # start_time = time.time()
                        # print(y_pred)
                        loss = loss_func(y_pred, y.squeeze(), reduction='mean')
                        reg_loss = self.get_regularization_loss()

                        total_loss = loss + reg_loss + self.aux_loss

                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        # nni.report_intermediate_result(total_loss.item())
                        total_loss.backward()
                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name == "auc_personal" or "binary_crossentropy":
                                    continue
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(metric_fun(
                                    y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()
            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / steps_per_epoch

            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                # eval_result = self.evaluate(val_x, val_y, batch_size)
                eval_result = self.evaluate_personal(val_x, val_y)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                logger.warning('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    if name == "auc_personal" or "binary_crossentropy":
                        continue
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name + \
                                    ": {0: .4f}".format(epoch_logs["val_" + name])
                logger.warning(eval_str)
            nni.report_intermediate_result(epoch_logs["val_auc_personal"])
            if epoch_logs["val_auc_personal"] >= best_val_score:
                best_val_score = epoch_logs["val_auc_personal"]
                best_model_params = model.state_dict()
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()

        return self.history, best_val_score, best_model_params

    def fit_BPR(self, logger, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None, lr=0.01):
        """

        :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
        :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
        :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
        :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`

        :return: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))

        else:
            val_x = []
            val_y = []
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        # optim = self.optim
        optim = torch.optim.Adam(self.parameters(), lr=lr)

        if self.gpus:
            logger.warning('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            logger.warning(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        # Train
        logger.warning("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        best_val_score = 0
        best_model_params = None
        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:

                        assert not len(x_train) % 2
                        i = 0
                        while i + 1 < len(y_train):
                            assert y_train[i] == 1, y_train[i]
                            assert y_train[i+1] == 0, y_train[i]
                            i += 2

                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()

                        y_pred = model(x).squeeze()

                        optim.zero_grad()
                        # start_time = time.time()
                        loss = loss_func(y_pred[::2], y_pred[1::2])
                        reg_loss = self.get_regularization_loss()

                        total_loss = loss + reg_loss + self.aux_loss

                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        nni.report_intermediate_result(total_loss.item())
                        total_loss.backward()
                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name == "auc_personal" or "binary_crossentropy":
                                    continue
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(metric_fun(
                                    y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()
            # logger.warning("one epoch loss set: ", myloss_set)
            # logger.warning("one epoch auc set: ", train_result["auc"])
            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / steps_per_epoch

            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                # eval_result = self.evaluate(val_x, val_y, batch_size)
                eval_result = self.evaluate_personal(val_x, val_y)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                logger.warning('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    if name == "auc_personal" or "binary_crossentropy":
                        continue
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name + \
                                    ": {0: .4f}".format(epoch_logs["val_" + name])
                logger.warning(eval_str)

            if epoch_logs["val_auc_personal"] >= best_val_score:
                best_val_score = epoch_logs["val_auc_personal"]
                best_model_params = model.state_dict()
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()

        return self.history, best_val_score, best_model_params

    def fit_BPR_new(self, logger, train_data_pos=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
                    validation_data=None, shuffle=True, callbacks=None, num_items=0, lr=0.01):

        """
        :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
        :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
        :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
        :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`

        :return: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """

        train_data_pos: pd.DataFrame  # (sample_num x (uid, iid, label=1))

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]
        else:
            val_x = []
            val_y = []

        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        # optim = self.optim
        optim = torch.optim.Adam(self.parameters(), lr=lr)

        if self.gpus:
            logger.warning('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            logger.warning(self.device)

        # 重排成  users_num x [interact_items]
        train_data_pos.reset_index(drop=True, inplace=True)
        interacn_matrix = train_data_pos.groupby(["userInt"]).newsInt.unique()
        ui_matrix = {uid: set(interacn_matrix[uid]) for uid in train_data_pos.userInt.unique()}
        print("构造数据集结束")
        # train_tensor_pos = torch.from_numpy(np.array(train_data_pos))

        sample_num = train_data_pos.shape[0]*2    # 1:1 采样，所以样本数是正样本的两倍
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        # Train
        logger.warning("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            sample_num, 0, steps_per_epoch))
        best_val_score = 0
        best_model_params = None

        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            '''
            x_pos = xxx
            sample
            x_neg = (x, y)
            concate(x_pos, x_neg)
            shuffle
            '''
            neg_data = pd.DataFrame(train_data_pos, copy=True)
            idx = 0
            for _ in range(len(train_data_pos)):
                uid = neg_data.iloc[idx, 0]
                while True:
                    neg_idx = random.randint(0, num_items-1)
                    if neg_idx not in ui_matrix[uid]:
                        neg_data.iloc[idx, 1] = neg_idx
                        idx += 1
                        break
            # neg_data.newsInt = pd.DataFrame(np.random.randint(0, item_size, size=(train_data_pos.shape[0], 1)))
            # data = pd.concat([train_data_pos, neg_data], axis=0)

            print("采样结束")
            train_tensor_data = Data.TensorDataset(torch.from_numpy(np.array(train_data_pos)),
                                                   torch.from_numpy(np.array(neg_data)))
            train_loader = DataLoader(dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_pos, x_neg) in t:

                        # assert not len(x_train) % 2
                        # i = 0
                        # while i + 1 < len(y_train):
                        #     assert y_train[i] == 1, y_train[i]
                        #     assert y_train[i+1] == 0, y_train[i]
                        #     i += 2
                        train_pos_len = x_pos.shape[0]
                        # x_train_neg = x_train.clone().detach()

                        # torch.tensor(x_train)
                        # x_train_neg = pd.DataFrame(x_train, copy=True)
                        # x_train_neg.label = 0
                        # idx = 0
                        # for _ in range(len(x_train)):
                        #     uid = x_train_neg.iloc[idx, 0]
                        #     while True:
                        #         neg_idx = random.randint(0, num_items-1)
                        #         if neg_idx not in ui_matrix[uid]:
                        #             x_train_neg.iloc[idx, 1] = neg_idx
                        #             idx += 1
                        #             break


                        # x_train_neg = torch.cat([x_train[:, 0:1], torch.randint(0, num_items, size=(train_pos_len, 1))], dim=1)
                        x = torch.cat([x_pos, x_neg], dim=0).to(self.device).float()
                        y = torch.ones((train_pos_len * 2, 1))
                        y[train_pos_len:, :] = 0
                        assert y[-1, :] == 0, "y有问题"
                        # print(y[-1, :])
                        y = y.to(self.device).float()
                        y_pred = model(x).squeeze()

                        optim.zero_grad()
                        # start_time = time.time()
                        # bpr_loss (pos, neg)
                        loss = loss_func(y_pred[:train_pos_len], y_pred[train_pos_len:])
                        reg_loss = self.get_regularization_loss()

                        total_loss = loss + reg_loss + self.aux_loss

                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        # nni.report_intermediate_result(total_loss.item())
                        total_loss.backward()
                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name == "auc_personal" or "binary_crossentropy":
                                    continue
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(metric_fun(
                                    y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()
            # logger.warning("one epoch loss set: ", myloss_set)
            # logger.warning("one epoch auc set: ", train_result["auc"])
            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / steps_per_epoch

            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                # eval_result = self.evaluate(val_x, val_y, batch_size)
                eval_result = self.evaluate_personal(val_x, val_y)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                logger.warning('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    if name == "auc_personal" or "binary_crossentropy":
                        continue
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name + \
                                    ": {0: .4f}".format(epoch_logs["val_" + name])
                logger.warning(eval_str)

            nni.report_intermediate_result(epoch_logs["val_auc_personal"])
            if epoch_logs["val_auc_personal"] >= best_val_score:
                best_val_score = epoch_logs["val_auc_personal"]
                best_model_params = model.state_dict()
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()

        return self.history, best_val_score, best_model_params



    def fit_SAUC(self, logger, x=None, train_data=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None, tau=0.02, items_data=None, items_num=16980):
        """
                train_data: DataFrame
        :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
        :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
        :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
        :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`

        :return: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        else:
            val_x = []
            val_y = []

        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            logger.warning('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            logger.warning(self.device)

        train_loader = DataLoader(
            dataset=x, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(x)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        # Train
        logger.warning("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(x), len(val_y), steps_per_epoch))

        best_val_score = 0
        best_model_params = None

        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            try:
                with tqdm(enumerate(train_loader), disable=verbose == 1) as t:
                    for _, (u, start, end) in t:
                        loss = 0.0
                        u = u.numpy()
                        start = start.numpy()
                        end = end.numpy()
                        # if len(u) == 0:
                        #     continue
                        for i in range(len(u)):
                            # assert not (end[i] - start[i]) % 2
                            pos_data = train_data.iloc[start[i]: end[i], :]  # userInt newsInt label
                            # todo: 等量负采样，再拼接，进行训练
                            # print("test")
                            u_pos = set(pos_data.newsInt.values)
                            neg_data = pd.DataFrame(pos_data, copy=True)
                            neg_data.label = 0
                            neg_data.reset_index(drop=True, inplace=True)
                            idx = 0
                            for _ in range(len(u_pos)):
                                while True:
                                    neg_idx = random.randint(0, items_num-1)
                                    if neg_idx not in u_pos:
                                        neg_data.loc[idx, "newsInt"] = neg_idx
                                        idx += 1
                                        break
                            # 替换负样本的物品特征
                            # if isinstance(items_data, pd.DataFrame):
                                # neg_data = pd.merge(neg_data[neg_data.columns.drop(["category", "subCategory", "imageNum"])],
                                #                     items_data[["newsInt", "category", "subCategory", "imageNum"]], how="left", on="newsInt")

                            data = pd.concat([pos_data, neg_data], axis=0)
                            del pos_data, neg_data
                            assert data.userInt.nunique() == 1
                            y = data.label
                            x = data.drop(columns="label")
                            del data
                            x = torch.tensor(x.astype(float).to_numpy()).to(self.device).float()
                            y = torch.tensor(y.astype(float).to_numpy()).to(self.device).float()
                            y_pred = model(x).squeeze()
                            if loss == 0:
                                loss = loss_func(y_pred, y.squeeze(), tau)
                            else:
                                loss += loss_func(y_pred, y.squeeze(), tau)
                            del x, y
                        loss /= len(u)
                        assert loss <= 1, f"smooth auc loss 必定小于1， 但是这里loss={loss}, len(u)={len(u)}"
                        optim.zero_grad()
                        reg_loss = self.get_regularization_loss()
                        total_loss = loss + reg_loss + self.aux_loss
                        # total_loss = loss

                        # nni.report_intermediate_result(total_loss.item())
                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward()
                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name == "auc_personal" or "binary_crossentropy":
                                    continue
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(metric_fun(
                                    y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()
            epoch_logs["loss"] = total_loss_epoch / steps_per_epoch

            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                # eval_result = self.evaluate(val_x, val_y, batch_size)
                eval_result = self.evaluate_personal(val_x, val_y)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                logger.warning('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    if name == "auc_personal" or "binary_crossentropy":
                        continue
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name + \
                                    ": {0: .4f}".format(epoch_logs["val_" + name])

                # for idx, (name, parameter) in enumerate(model.named_parameters()):
                #     logger.warning(name, ":", parameter)
                #     if idx > 4:
                #         break
                logger.warning(eval_str)
            nni.report_intermediate_result(epoch_logs["val_auc_personal"])
            if epoch_logs["val_auc_personal"] >= best_val_score:
                best_val_score = epoch_logs["val_auc_personal"]
                best_model_params = model.state_dict()

            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()

        return self.history, best_val_score, best_model_params

    def fit_SAUC_Lambda(self, logger, x=None, train_data=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None, tau=0.02, items_data=None, items_num=16980,lr=0.01):
        """
                train_data: DataFrame
        :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
        :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
        :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
        :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`

        :return: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        else:
            val_x = []
            val_y = []

        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        # optim = self.optim
        optim = torch.optim.Adam(self.parameters(), lr=lr)

        if self.gpus:
            logger.warning('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            logger.warning(self.device)

        train_loader = DataLoader(
            dataset=x, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(x)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        # Train
        logger.warning("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(len(x), len(val_y), steps_per_epoch))

        best_val_score = 0
        best_model_params = None

        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            total_sauc_loss = 0
            train_result = {}
            try:
                with tqdm(enumerate(train_loader), disable=verbose == 1) as t:
                    for _, (u, start, end) in t:
                        mean_loss, sum_loss = 0.0, 0.0
                        u = u.numpy()
                        start = start.numpy()
                        end = end.numpy()
                        if len(u) == 0:
                            continue
                        for i in range(len(u)):
                            pos_data = train_data.iloc[start[i]: end[i], :]  # userInt newsInt label
                            u_pos = set(pos_data.newsInt.values)
                            neg_data = pd.DataFrame(pos_data, copy=True)
                            neg_data.label = 0
                            neg_data.reset_index(drop=True, inplace=True)
                            idx = 0
                            for _ in range(len(u_pos)):
                                while True:
                                    neg_idx = random.randint(0, items_num - 1)
                                    if neg_idx not in u_pos:
                                        neg_data.loc[idx, "newsInt"] = neg_idx
                                        idx += 1
                                        break

                            x_pos = torch.tensor(pos_data.drop(columns="label").astype(float).to_numpy()).to(self.device).float()
                            x_neg = torch.tensor(neg_data.drop(columns="label").astype(float).to_numpy()).to(self.device).float()

                            pos_pred = model(x_pos)
                            neg_pred = model(x_neg)
                            mean_loss_single, sum_loss_single, sauc_loss = loss_func(pos_pred, neg_pred, tau=tau)
                            mean_loss += mean_loss_single
                            sum_loss += sum_loss_single
                            total_sauc_loss += sauc_loss.item()
                        mean_loss /= len(u)
                        sum_loss /= len(u)
                        # assert loss <= 1, f"smooth auc loss 必定小于1， 但是这里loss={loss}, len(u)={len(u)}"
                        optim.zero_grad()
                        reg_loss = self.get_regularization_loss()
                        total_loss = sum_loss + reg_loss + self.aux_loss
                        # total_loss = mean_loss + reg_loss + self.aux_loss
                        total_sauc_loss /= len(u)
                        # total_loss = loss

                        # nni.report_intermediate_result(total_loss.item())
                        loss_epoch += mean_loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward()
                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name == "auc_personal" or "binary_crossentropy":
                                    continue
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(metric_fun(y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()
            epoch_logs["total_loss"] = total_loss_epoch / steps_per_epoch
            epoch_logs["loss"] = loss_epoch / steps_per_epoch
            epoch_logs['sauc_loss'] = total_sauc_loss / steps_per_epoch

            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                # eval_result = self.evaluate(val_x, val_y, batch_size)
                eval_result = self.evaluate_personal(val_x, val_y)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                logger.warning('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - total loss: {1: .4f} - loss: {2: .4f} - sauc loss: {3: .4f}".format(epoch_time, epoch_logs["total_loss"], epoch_logs['loss'], epoch_logs['sauc_loss'])

                for name in self.metrics:
                    if name == "auc_personal" or "binary_crossentropy":
                        continue
                    eval_str += " - " + name + ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name + ": {0: .4f}".format(epoch_logs["val_" + name])

                # for idx, (name, parameter) in enumerate(model.named_parameters()):
                #     logger.warning(name, ":", parameter)
                #     if idx > 4:
                #         break
                logger.warning(eval_str)
            nni.report_intermediate_result(epoch_logs["val_auc_personal"])
            if epoch_logs["val_auc_personal"] >= best_val_score:
                best_val_score = epoch_logs["val_auc_personal"]
                best_model_params = model.state_dict()

            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training or epoch_logs["val_auc_personal"] <= 0.1:
                break

        callbacks.on_train_end()

        return self.history, best_val_score, best_model_params

    def fit_SAUC_Lambda_new(self, logger, x=None, train_data=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None, tau=0.02, items_data=None, items_num=16980,lr=0.01):
        """
                train_data: DataFrame
        :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
        :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
        :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
        :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`

        :return: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        train_data_pos: pd.DataFrame  # (sample_num x (uid, iid, label=1))

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        else:
            val_x = []
            val_y = []

        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        # optim = self.optim
        optim = torch.optim.Adam(self.parameters(), lr=lr)

        if self.gpus:
            logger.warning('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            logger.warning(self.device)

        train_loader = DataLoader(dataset=x, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(x)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        # Train
        logger.warning("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(len(x), len(val_y), steps_per_epoch))

        best_val_score = 0
        best_model_params = None

        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            total_sauc_loss = 0
            train_result = {}
            try:
                with tqdm(enumerate(train_loader), disable=verbose == 1) as t:
                    for _, (u, start, end) in t:
                        mean_loss, sum_loss = 0.0, 0.0
                        u = u.numpy()
                        start = start.numpy()
                        end = end.numpy()
                        if len(u) == 0:
                            continue
                        for i in range(len(u)):
                            pos_data = train_data.iloc[start[i]: end[i], :]  # userInt newsInt label
                            u_pos = set(pos_data.newsInt.values)
                            neg_data = pd.DataFrame(pos_data, copy=True)
                            neg_data.label = 0
                            neg_data.reset_index(drop=True, inplace=True)
                            idx = 0
                            for _ in range(len(u_pos)):
                                while True:
                                    neg_idx = random.randint(0, items_num - 1)
                                    if neg_idx not in u_pos:
                                        neg_data.loc[idx, "newsInt"] = neg_idx
                                        idx += 1
                                        break

                            x_pos = torch.tensor(pos_data.drop(columns="label").astype(float).to_numpy()).to(self.device).float()
                            x_neg = torch.tensor(neg_data.drop(columns="label").astype(float).to_numpy()).to(self.device).float()

                            pos_pred = model(x_pos)
                            neg_pred = model(x_neg)
                            mean_loss_single, sum_loss_single, sauc_loss = loss_func(pos_pred, neg_pred, tau=tau)
                            mean_loss += mean_loss_single
                            sum_loss += sum_loss_single
                            total_sauc_loss += sauc_loss.item()
                        mean_loss /= len(u)
                        sum_loss /= len(u)
                        # assert loss <= 1, f"smooth auc loss 必定小于1， 但是这里loss={loss}, len(u)={len(u)}"
                        optim.zero_grad()
                        reg_loss = self.get_regularization_loss()
                        total_loss = sum_loss + reg_loss + self.aux_loss
                        total_sauc_loss /= len(u)
                        # total_loss = loss

                        # nni.report_intermediate_result(total_loss.item())
                        loss_epoch += mean_loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward()
                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name == "auc_personal" or "binary_crossentropy":
                                    continue
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(metric_fun(y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()
            epoch_logs["total_loss"] = total_loss_epoch / steps_per_epoch
            epoch_logs["loss"] = loss_epoch / steps_per_epoch
            epoch_logs["sauc_loss"] = total_sauc_loss / steps_per_epoch

            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                # eval_result = self.evaluate(val_x, val_y, batch_size)
                eval_result = self.evaluate_personal(val_x, val_y)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                logger.warning('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - total loss: {1: .4f} - loss: {2: .4f}".format(epoch_time, epoch_logs["total_loss"], epoch_logs['loss'], epoch_logs['sauc_loss'])

                for name in self.metrics:
                    if name == "auc_personal" or "binary_crossentropy":
                        continue
                    eval_str += " - " + name + ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name + ": {0: .4f}".format(epoch_logs["val_" + name])

                # for idx, (name, parameter) in enumerate(model.named_parameters()):
                #     logger.warning(name, ":", parameter)
                #     if idx > 4:
                #         break
                logger.warning(eval_str)
            nni.report_intermediate_result(epoch_logs["val_auc_personal"])
            if epoch_logs["val_auc_personal"] >= best_val_score:
                best_val_score = epoch_logs["val_auc_personal"]
                best_model_params = model.state_dict()

            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training or epoch_logs["val_auc_personal"] <= 0.1:
                break

        callbacks.on_train_end()

        return self.history, best_val_score, best_model_params


    def evaluate(self, x, y, batch_size=256):
        """
        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans = self.predict(x, batch_size)
        assert not len(pred_ans) % 101
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(y, pred_ans)
        return eval_result

    def evaluate_personal(self, x, y, batch_size=101):
        """

        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        i = 0
        while i < len(y):
            assert y[i] == 1
            assert np.sum(y[i:i + 101]) == 1
            i += 101
        pred_ans, auc_personal = self.predict_personal(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            if name == "auc_personal":
                continue
            eval_result[name] = metric_fun(y, pred_ans)
        eval_result["auc_personal"] = np.mean(auc_personal)
        return eval_result

    def test_personal(self, x, y, batch_size=101):
        """
        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        i = 0
        while i < len(y):
            assert y[i] == 1
            i += 101
        pred_ans, auc_personal, map, mrr, NDCG, recall = self.test_predict_personal(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            if name == "auc_personal":
                continue
            eval_result[name] = metric_fun(y, pred_ans)
        eval_result["auc_personal"] = auc_personal
        eval_result["mrr"] = mrr
        eval_result["map 2 4 6 8 10"] = map
        eval_result["NDCG"] = NDCG
        eval_result["recall 2 4 6 8 10"] = recall
        return eval_result

    def predict(self, x, batch_size=256):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()

                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")

    def predict_personal(self, x, batch_size=101):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        auc_personal = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()

                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)
                count = np.sum([y_pred[1:] < y_pred[0]])
                assert count <= 100
                auc_personal.append(count/100)
        return np.concatenate(pred_ans).astype("float64"), auc_personal


    def test_predict_personal(self, x, batch_size=101):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        auc_personal = []
        map = []
        mrr = []
        NDCG = []
        recall = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()

                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)
                count = np.sum([y_pred[1:] < y_pred[0]])
                assert count <= 100
                auc_personal.append(count/100)
                temp = self.map_recall_at_k_multileveltobinary(np.array([1] + [0]*100), y_pred.reshape(-1), [2, 4, 6, 8, 10])
                recall.append(temp[0])
                map.append(temp[1])
                mrr.append(temp[2])
                NDCG.append(self.normalized_discounted_cumulative_gain_matrix(np.array([1] + [0]*100), y_pred.reshape(-1), 10))
        return np.concatenate(pred_ans).astype("float64"), np.mean(auc_personal), np.mean(map, axis=0), np.mean(mrr), np.mean(NDCG, axis=0), np.mean(recall, axis=0)
        # return np.concatenate(pred_ans).astype("float64"), np.mean(auc_personal)
    def AP_MRR(self, binary_gt, y_pred):
        pred_rank = y_pred.argsort()[::-1]
        binary_gt_rank = np.array([binary_gt[pred_rank[i]] for i in range(len(pred_rank))])
        mrr_value = max(binary_gt_rank / np.arange(1, len(binary_gt_rank) + 1))
        ap = (binary_gt_rank * np.cumsum(binary_gt_rank) / (1 + np.arange(len(binary_gt_rank))))
        return ap[np.nonzero(ap)].mean(), mrr_value

    def map_recall_at_k_multileveltobinary(self, y_true, y_pred, Ks):
        '''
        recall_value, map_value, mrr_value

        :param y_true:
        :param y_pred:
        :param Ks:    K的list
        :return:
        '''

        pred_rank = y_pred.argsort()[::-1]
        recall_value = np.zeros(len(Ks))
        map_value = np.zeros(len(Ks))
        true_rank = np.array([y_true[pred_rank[i]] for i in range(len(pred_rank))])
        mrr_value = max(true_rank / np.arange(1, len(true_rank) + 1))
        pos_items = np.sum(true_rank)
        for i in range(len(Ks)):
            cut_off = min(Ks[i], len(y_true))
            recall_value[i] = np.sum(true_rank[:cut_off]) / pos_items
            p_value = (true_rank[:cut_off] * np.cumsum(true_rank[:cut_off])) / (1 + np.arange(cut_off))
            p_value = p_value[np.nonzero(p_value)].mean() if p_value[np.nonzero(p_value)].size > 0 else 0
            map_value[i] = p_value
        return recall_value, map_value, mrr_value

    def normalized_discounted_cumulative_gain_matrix(self, label: np.ndarray, pred: np.ndarray, K: int):
        '''
        返回[1:k + 1] 所有的NDCG@K.
        :param K:
        :param label:
        :param pred:
        :return:
        '''
        assert len(label) == len(pred)
        dcg_matrix = np.zeros(K)
        prank = pred.argsort()[::-1]
        dcg_list = [(2 ** label[prank[r]] - 1) / math.log2(r + 2) for r in range(min(K, len(label)))]
        for r in range(1, K + 1):
            dcg_matrix[r - 1] = sum(dcg_list[:min(r, len(label))])
        return dcg_matrix / self.ideal_discounted_cumulative_gain_matrix(K, label)

    def ideal_discounted_cumulative_gain_matrix(self, K: int, label: np.ndarray):
        lrank = label.argsort()[::-1]
        idcg_matrix = np.zeros(K)
        idcg_list = [(2 ** label[lrank[r]] - 1) / math.log2(r + 2) for r in range(min(K, len(label)))]
        for r in range(1, K + 1):
            idcg_matrix[r - 1] = sum(idcg_list[:min(r, len(label))])
        return idcg_matrix

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")

        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()
        ) for feat in sparse_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      varlen_sparse_feature_columns)
        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                               varlen_sparse_feature_columns, self.device)

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(
            map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self, ):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def add_auxiliary_loss(self, aux_loss, alpha):
        self.aux_loss = aux_loss * alpha

    def compile(self,
                loss=None,
                metrics=None,
                ):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        """
        self.metrics_names = ["loss"]
        # self.optim = self._get_optim(optimizer)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def _get_optim(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=self.lr)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters(), lr=self.lr)  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters())  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters())
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            elif loss == "smooth_auc_loss":
                loss_func = self.sal
            elif loss == "smooth_auc_loss_lambda":
                loss_func = self.sall
            elif loss == "bpr":
                loss_func = self.bpr
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true,
                        y_pred,
                        eps,
                        normalize,
                        sample_weight,
                        labels)

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1, 0))
                if metric == "auc_personal":
                    metrics_[metric] = "pass"
                self.metrics_names.append(metric)
        return metrics_

    def _in_multi_worker_mode(self):
        # used for EarlyStopping in tf1.15
        return None

    @property
    def embedding_size(self, ):
        feature_columns = self.dnn_feature_columns
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        embedding_size_set = set([feat.embedding_dim for feat in sparse_feature_columns])
        if len(embedding_size_set) > 1:
            raise ValueError("embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!")
        return list(embedding_size_set)[0]

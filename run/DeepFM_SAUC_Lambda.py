import logging
import random
from time import time
from datetime import datetime
import pandas as pd
import torch
import nni
import sys
from tensorflow.keras.callbacks import EarlyStopping
import os
sys.path.append("..")
from utils import logCof
# from models import MF, SmoothAUCLoss, BPR
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DeepFM

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
current_time = datetime.now().strftime('%Y%m%d%H%M%S')


def main(args):
    # items_data = pd.read_csv(os.path.join(args["datadir"], "items_info.csv"))

    torch.manual_seed(args["seed"])
    random.seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])

    sparse_features = {"userInt": 5560, "newsInt": 17000}
    # dense_features = [str(x) for x in range(100)]
    dense_features = []

    fixlen_feature_columns = [SparseFeat(feat, sparse_features[feat], embedding_dim=args["embedding_dim"]) for feat in sparse_features] \
                             + [DenseFeat(feat, 1, ) for feat in dense_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    model = DeepFM(linear_feature_columns=linear_feature_columns,
                   dnn_feature_columns=dnn_feature_columns,
                   l2_reg_embedding=args["l2_reg_embedding"],
                   l2_reg_dnn=args["l2_reg_dnn"],
                   device=device,
                   dnn_hidden_units=(args["layer_size1"], args["layer_size2"]),  #
                   # dnn_hidden_units=(),  #
                   dnn_dropout=args["dropout"],
                   dnn_use_bn=True,
                   # use_fm=False,
                   )
    model.compile('smooth_auc_loss_lambda', metrics=["binary_crossentropy", 'auc_personal'])

    if not args["only_test"]:

        train_data = pd.read_csv(os.path.join(args["datadir"], "train_data1.csv"))
        val_data = pd.read_csv(os.path.join(args["datadir"], "val_data.csv"))
        train3 = pd.read_pickle(os.path.join(args["datadir"], "train3.pickle"))["train_data3_user_list"]
        train_data.columns = ["userInt", "newsInt", "label"]
        val_data.columns = ["userInt", "newsInt", "label"]

        callback = EarlyStopping(monitor="val_auc_personal", patience=10, verbose=1, mode="max") # patience=10
        logger.warning("*****************************        Start Training     ******************************")
        history, best_val_score, best_model_params = model.fit_SAUC_Lambda(logger, train3, train_data[[name for name in feature_names] + ["label"]],
                                              batch_size=args["batch_size"], epochs=args["epochs"], verbose=1,
                                              validation_data=[{name: val_data.drop(columns=["label"])[name] for name in feature_names}, val_data.label.values],
                                              callbacks=[callback],
                                              shuffle=True, tau=args["tau"], lr=args["lr"])
        nni.report_final_result(best_val_score)
        # save model
        dirname = os.path.dirname(os.path.abspath(args["model_path"]))
        os.makedirs(dirname, exist_ok=True)
        model_name = args["project_name"] + "_" + current_time + "_lr_" + str(args["lr"]) + "_tau_" + str(args["tau"]) + "_" + str(best_val_score)[:8] + ".pt"
        torch.save(best_model_params, os.path.join(dirname, model_name))
        # print("*****************************        Start Testing     ******************************")
        logger.warning("*****************************        Start Testing     ******************************")
        test_data = pd.read_csv(os.path.join(args["datadir"], "test_data.csv"))
        test_data.columns = ["userInt", "newsInt", "label"]
        eval_result = model.test_personal({name: test_data.drop(columns=["label"])[name] for name in feature_names}, test_data.label.values)
        for name, values in eval_result.items():
            # print(name, values)
            logger.warning(name + ' ' + str(values))
    else:
        print('**'*30 + 'Loading best model from local' + '**'*30)
        test_model_name = "CiteULike_DeepFM_SAUC_20220927173548_tau_0.02_0.900003.pt"
        test_data = pd.read_csv(os.path.join(args["datadir"], "test_data.csv"))
        test_data.columns = ["userInt", "newsInt", "label"]
        model_dict = model.load_state_dict(torch.load(os.path.join("../saved_models", test_model_name)))
        print("test on model:   ", test_model_name)
        eval_result = model.test_personal({name: test_data.drop(columns=["label"])[name] for name in feature_names}, test_data.label.values)
        for name, values in eval_result.items():
            print(name, values)


def get_default_parameters():
    # 要调参的参数
    params = \
        {
            "lr": 0.05,
            "dropout": 0.35,
            "embedding_dim": 8,
            "layer_size1": 32,
            "layer_size2": 8,
            "batch_size": 1000,
            "l2_reg_embedding": 0.0001,
            "l2_reg_dnn": 0.00001,
            "tau": 0.05
        }
    return params


if __name__ == '__main__':
    # 一些默认参数
    params = \
        {
            "current_time": current_time,
            "project_name": "CiteULike_DeepFM_SAUC_Lambda",
            "datadir": "/data/lfyuan/Datasets/CiteULike/",
            "dataset": 'CiteULike',
            "seed": 0,
            "cuda": 0,
            "tau": 0.1,
            # train
            "lr": 0.005,
            "dropout": 0.9,
            "batch_size": 2000,
            "epochs": 3000,
            # save
            "model_path": "../saved_models/xxx.pt",
            # test
            "only_test": False
        }

    # device = "cpu"
    # use_cuda = True
    # if use_cuda and torch.cuda.is_available():
    #     print("cuda ready...")
    #     device = "cuda:" + str(params["cuda"])

    device = "cuda:" + str(params["cuda"])

    logger = logging.getLogger(params["project_name"])
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        params.update(get_default_parameters())
        params.update(tuner_params)
        log_file_name = params["project_name"] + "_" + current_time + "_" + str(params["lr"]) + "_" + str(params["tau"]) + ".log"
        logger = logCof(logger, "../log/", log_file_name)
        logger.info(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise

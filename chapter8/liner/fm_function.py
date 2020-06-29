
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold

from .config import *
from .metrics import gini_norm
from .DataReader import FeatureDictionary, DataParser
sys.path.append("..")
from .DeepFM import DeepFM

gini_scorer = make_scorer(gini_norm, greater_is_better=True, needs_proba=True)


def _load_data():




    dfTrain = pd.read_csv(TRAIN_FILE,header= 0,sep= '\t' )
    dfTest = pd.read_csv(TEST_FILE,header= 0,sep='\t')
    # print('######################')
    # print (dfTrain)
    # print ('######################')
    # print(dfTrain.info())
    # print('######################')
    #print(dfTest,dfTest.info())

    def preprocess(df):
        cols = [c for c in df.columns if c not in ["id", "target"]]
        return df

    dfTrain = preprocess(dfTrain)
    dfTest = preprocess(dfTest)


    #print('######################')
    #print (dfTrain)
    #print ('######################')
    #print(dfTrain.info())
    #print('######################')


    cols = [c for c in dfTrain.columns if c not in ["id", "target"]]
    cols = [c for c in cols if (not c in IGNORE_COLS)]


    print(dfTrain)
    X_train = dfTrain[cols].values
    y_train = dfTrain["target"].values
    X_test = dfTest[cols].values
    ids_test = dfTest["id"].values
    cat_features_indices = [i for i,c in enumerate(cols) if c in CATEGORICAL_COLS]

    # print('##################')
    # print(dfTrain)
    #
    # print('##################')
    # print (X_train)
    #
    # print('##################')
    # print(y_train)



    return dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices


def _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params):
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           numeric_cols=NUMERIC_COLS,
                           ignore_cols=IGNORE_COLS)



    data_parser = DataParser(feat_dict=fd)







    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    Xi_test, Xv_test, ids_test = data_parser.parse(df=dfTest)

    print('y_train = ',y_train[0])
    print('x i input = ',Xi_test[0])
    print('x v input = ',Xv_test[0])
    print('ids = ',ids_test[0])

    #
    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0])

    y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
    y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)
    _get = lambda x, l: [x[i] for i in l]
    gini_results_cv = np.zeros(len(folds), dtype=float)
    gini_results_epoch_train = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    gini_results_epoch_valid = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)


        dfm = DeepFM(**dfm_params)
        dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)

        y_train_meta[valid_idx,0] = dfm.predict(Xi_valid_, Xv_valid_)
        y_test_meta[:,0] += dfm.predict(Xi_test, Xv_test)

        gini_results_cv[i] = gini_norm(y_valid_, y_train_meta[valid_idx])
        gini_results_epoch_train[i] = dfm.train_result
        gini_results_epoch_valid[i] = dfm.valid_result

    y_test_meta /= float(len(folds))

    # save result
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "DeepFM"
    elif dfm_params["use_fm"]:
        clf_str = "FM"
    elif dfm_params["use_deep"]:
        clf_str = "DNN"
    print("%s: %.5f (%.5f)"%(clf_str, gini_results_cv.mean(), gini_results_cv.std()))
    filename = "%s_Mean%.5f_Std%.5f.csv"%(clf_str, gini_results_cv.mean(), gini_results_cv.std())
    _make_submission(ids_test, y_test_meta, filename)

    _plot_fig(gini_results_epoch_train, gini_results_epoch_valid, clf_str)

    return y_train_meta, y_test_meta


def _make_submission(ids, y_pred, filename="submission.csv"):
    pd.DataFrame({"id": ids, "target": y_pred.flatten()}).to_csv(
        os.path.join(SUB_DIR, filename), index=False, float_format="%.5f")


def _plot_fig(train_results, valid_results, model_name):
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_results.shape[1]+1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d"%(i+1))
        legends.append("valid-%d"%(i+1))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s"%model_name)
    plt.legend(legends)
    plt.savefig("./fig/%s.png"%model_name)
    plt.close()



# main process
def fm_function(key,params):

    # load data
    print ('load data')
    dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = _load_data()

    #print (cat_features_indices)

    #return 0

    # folds
    folds = list(StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True,
                                 random_state=RANDOM_SEED).split(X_train, y_train))


    # ------------------ DeepFM Model ------------------

    dfm_params = {
        "embedding_size": 8,
        "dropout_fm": [1.0, 1.0],
        "deep_layers": [32, 32],
        "dropout_deep": [0.8, 0.8, 0.8],
        "deep_layers_activation": tf.nn.relu,
        "epoch": 30,
        "batch_size": 1024,
        "learning_rate": 0.01,
        "optimizer_type": "adam",
        "batch_norm": 1,
        "batch_norm_decay": 0.995,
        "l2_reg": 0.01,
        "verbose": True,
        "eval_metric": gini_norm,
        "random_seed": RANDOM_SEED
    }

    for i in params :
        if i in dfm_params :
            dfm_params[i] = params[i]
        else:
            print(i ,' is not a param of function , check your input')


    #print('train data input = ',dfTrain)

    if 'DeepFM' in key :
        print('DeepFM is begin')
        for i, (train_idx, valid_idx) in enumerate(folds):
            print ('folds = ',i, train_idx,valid_idx)
        #return 0
        y_train_dfm, y_test_dfm = _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params)
        print ('DeepFM is done' )

    if 'FM' in key :
        # ------------------ FM Model ------------------
        fm_params = dfm_params.copy()

        print('FM is begin')
        y_train_fm, y_test_fm = _run_base_model_dfm(dfTrain, dfTest, folds, fm_params)
        print('FM is done')

    if 'DNN' in key :
        # ------------------ DNN Model ------------------
        dnn_params = dfm_params.copy()
        print ('start to run dnn ')
        print('DNN is begin')
        y_train_dnn, y_test_dnn = _run_base_model_dfm(dfTrain, dfTest, folds, dnn_params)
        print('DNN is done')


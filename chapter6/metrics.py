"""
@Time ： 2020/6/24 13:53
@Auth ： sunmingzhu
@File ：metrics.py
@IDE ：PyCharm

"""

import numpy as np
from sklearn.metrics import *


# 准确率 accuracy=(TP+TN)/(TP+TN+FP+FN)
def compute_accuracy():
    y_pred = [0, 2, 1, 3]
    y_true = [0, 1, 2, 3]
    print(accuracy_score(y_true, y_pred))
    print(accuracy_score(y_true, y_pred, normalize=False))  # normalize默认为false，为true时，返回分类正确的样本数量
    # 在具有二元标签指示符的多标签分类案例中
    print(accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2))))


# 精确率 precision=TP/(TP+FP)
def compute_precision():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]
    # Macro在计算均值时使每个类别具有相同的权重，最后结果是每个类别的指标的算术平均值
    print(precision_score(y_true, y_pred, average='macro'))
    # micro在计算均值时给每个类别下的每个样本相同的权重，将所有样本合在一起计算各个指标
    print(precision_score(y_true, y_pred, average='micro'))
    print(precision_score(y_true, y_pred, average='weighted'))
    # 当average参数为None时，得到的结果是每个类别的precision
    print(precision_score(y_true, y_pred, average=None))


def compute_recall():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]

    print(recall_score(y_true, y_pred, average='macro'))
    print(recall_score(y_true, y_pred, average='micro'))
    print(recall_score(y_true, y_pred, average='weighted'))
    print(recall_score(y_true, y_pred, average=None))


# P-R曲线，精准率precision和召回率recall曲线，以recall作为横坐标轴，precision作为纵坐标轴，主要是针对二分类
def compute_precision_recall():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    print("precision:" + str(precision))
    print("recall:" + str(recall))
    print("thresholds:" + str(thresholds))


# F1 = 2 * (precision * recall) / (precision + recall)
def compute_f1_score():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]
    print(f1_score(y_true, y_pred, average='macro'))  # 0.26666666666666666
    print(f1_score(y_true, y_pred, average='micro'))  # 0.3333333333333333
    print(f1_score(y_true, y_pred, average='weighted'))  # 0.26666666666666666
    print(f1_score(y_true, y_pred, average=None))  # [0.8 0.  0. ]


def DCG(label_list):
    dcgsum = 0
    for i in range(len(label_list)):
        dcg = (2 ** label_list[i] - 1) / np.math.log(i + 2, 2)
        dcgsum += dcg
    return dcgsum


def NDCG(label_list):
    dcg = DCG(label_list)
    ideal_list = sorted(label_list, reverse=True)
    ideal_dcg = DCG(ideal_list)
    if ideal_dcg == 0:
        return 0
    return dcg / ideal_dcg


def results_reshape(Query, Documents, targets, predicts):
    qq_dict = {}
    for i in range(len(targets)):
        query = Query[i]
        if query not in qq_dict.keys():
            qq_dict[query] = []
            qq_dict[query].append((Documents[i], targets[i], predicts[i]))
        else:
            qq_dict[query].append(
                (Documents[i], targets[i], predicts[i]))
    return qq_dict


def retrival_evaluate(qq_dict, k):
    # Calculate all metrics for comparing performance
    MAP, MRR = 0, 0
    correct = 0
    Precision_k = 0
    NDCG_k = 0
    has2qnum = 0
    for s1 in qq_dict.keys():
        p, AP = 0, 0
        MRR_check = False
        precision = 0
        label_list = []
        qq_dict[s1] = sorted(qq_dict[s1], key=lambda x: x[-1], reverse=False)
        (quest, label, prob) = qq_dict[s1][0]  # ans,
        if label != 0.0:
            correct += 1
        for idx, (quest, label, prob) in enumerate(qq_dict[s1]):  # ans,
            if label != 0.0:
                if not MRR_check:
                    MRR += 1 / (idx + 1)
                    MRR_check = True
                p += 1
                AP += p / (idx + 1)
            if idx < k:
                label_list.append(label)
                if label != 0.0:
                    precision += 1
        if p > 0:
            has2qnum += 1
        AP /= (p + 1e-07)
        MAP += AP
        Precision_k += precision / k
        NDCG_k += NDCG(label_list)

    num_questions = len(qq_dict.keys())
    print('question number：', num_questions)
    MAP /= has2qnum
    MRR /= has2qnum
    correct /= has2qnum
    Precision_k /= has2qnum
    NDCG_k /= num_questions

    print('MAP:', MAP, ';  MRR:', MRR, ';  Top 1 accuracy:', correct, ';  P@k:', Precision_k, ';  NDCG@k:', NDCG_k)
    return MAP, MRR, correct, Precision_k, NDCG_k

if __name__ == "__main__":
    compute_precision_recall()

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
# 计算准确率(Accurary) 、 精确率（Precision）、召回率（Recall）、F1-Score
# 分别计算TP FN FP TN


# 计算TP
def cal_tp(data, tag):
    TP = 0
    for line in data:
        if line == '\n':
            continue
        line = line.strip('\n').split('\t')
        actual_value = line[-2]
        predicted_value = line[-1]

        if actual_value[0] == tag and actual_value == predicted_value:
            TP += 1
    return TP


# 计算FN
def cal_fn(data, tag):
    FN = 0
    for line in data:
        if line == '\n':
            continue
        line = line.strip('\n').split('\t')
        actual_value = line[-2]
        predicted_value = line[-1]

        if actual_value[0] == tag and actual_value != predicted_value:
            FN += 1
    return FN


# 计算FP
def cal_fp(data, tag):
    FP = 0
    for line in data:
        if line == '\n':
            continue
        line = line.strip('\n').split('\t')
        actual_value = line[-2]
        predicted_value = line[-1]

        if predicted_value[0] == tag and actual_value != predicted_value:
            FP += 1
    return FP


# 计算TN
def cal_tn(data, tag):
    TN = 0
    for line in data:
        if line == '\n':
            continue
        line = line.strip('\n').split('\t')
        actual_value = line[-2]
        predicted_value = line[-1]

        if predicted_value[0] != tag and actual_value[0] != tag:
            TN += 1
    return TN


def cal_main():
    with open('test_log.txt', 'r', encoding='UTF-8') as fw:
        data = fw.readlines()

    # tag = 'B_LOCATION'
    tags = ['B', 'I', 'E', 'S', 'O']

    macro_acc = []
    macro_pre = []
    macro_rec = []
    macro_f1 = []

    for tag in tags:
        TP = cal_tp(data, tag)
        FN = cal_fn(data, tag)
        FP = cal_fp(data, tag)
        TN = cal_tn(data, tag)

        precision = 0 if TP + FP == 0 else TP / (TP+FP)
        recall = 0 if TP + FN == 0 else TP / (TP+FN)
        accurary = 0 if TP+TN+FN+FP == 0 else (TP+TN) / (TP+TN+FN+FP)
        f1 = 0 if precision + recall == 0 else 2*precision*recall/(precision+recall)

        print(tag, ', Accuracy', round(accurary, 3),
              ', Precision', round(precision, 3),
              ', Recall', round(recall, 3),
              ', F1 Score', round(f1, 3))
        macro_acc.append(accurary)
        macro_pre.append(precision)
        macro_rec.append(recall)
        macro_f1.append(f1)

    macro_pre_ = np.expand_dims(np.array(macro_pre), axis=1)
    macro_rec_ = np.expand_dims(np.array(macro_rec), axis=1)
    macro_f1_ = np.expand_dims(np.array(macro_f1), axis=1)

    overall = np.concatenate([macro_rec_, macro_pre_, macro_f1_], axis=1)
    overall = pd.DataFrame(overall)
    overall.to_csv('results.csv', header=None, index=None)

    macro_acc = sum(macro_acc) / len(macro_acc)
    macro_pre = sum(macro_pre) / len(macro_pre)
    macro_rec = sum(macro_rec) / len(macro_rec)
    macro_f1 = sum(macro_f1) / len(macro_f1)

    print('Macro Accuracy', round(macro_acc, 3),
          ', Macro Precision', round(macro_pre, 3),
          ', Macro Recall', round(macro_rec, 3),
          ', Macro F1 Score', round(macro_f1, 3))


if __name__ == '__main__':
    cal_main()

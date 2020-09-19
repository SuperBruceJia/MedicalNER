# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


# 计算TP
def cal_tp(data, tag):
    TP = 0
    for line in data:
        if line == '\n':
            continue
        line = line.strip('\n').split('\t')
        actual_value = line[-2][2:]
        predicted_value = line[-1][2:]
        if actual_value == tag and actual_value == predicted_value:
            TP += 1
    return TP


# 计算FN
def cal_fn(data, tag):
    FN = 0
    for line in data:
        if line == '\n':
            continue
        line = line.strip('\n').split('\t')
        actual_value = line[-2][2:]
        predicted_value = line[-1][2:]
        if actual_value == tag and actual_value != predicted_value:
            FN += 1
    return FN


# 计算FP
def cal_fp(data, tag):
    FP = 0
    for line in data:
        if line == '\n':
            continue
        line = line.strip('\n').split('\t')
        actual_value = line[-2][2:]
        predicted_value = line[-1][2:]
        if predicted_value == tag and actual_value != predicted_value:
            FP += 1
    return FP


# 计算TN
def cal_tn(data, tag):
    TN = 0
    for line in data:
        if line == '\n':
            continue
        line = line.strip('\n').split('\t')
        actual_value = line[-2][2:]
        predicted_value = line[-1][2:]
        if predicted_value != tag and actual_value != tag:
            TN += 1
    return TN


def cal_main():
    with open('test_log.txt', 'r', encoding='UTF-8') as fw:
        data = fw.readlines()

    tags = ['E95f2a617',
            'E320ca3f6',
            'E340ca71c',
            'E1ceb2bd7',
            'E1deb2d6a',
            'E370cabd5',
            'E360caa42',
            'E310ca263',
            'E300ca0d0',
            'E18eb258b',
            'E3c0cb3b4',
            'E1beb2a44',
            'E3d0cb547',
            'E8ff29ca5',
            'E330ca589',
            'E1eeb2efd',
            'E17eb23f8',
            'E94f2a484']

    TP = 0
    FN = 0
    FP = 0
    TN = 0

    for tag in tags:
        TP += cal_tp(data, tag)
        FN += cal_fn(data, tag)
        FP += cal_fp(data, tag)
        TN += cal_tn(data, tag)

    precision = 0 if TP + FP == 0 else TP / (TP + FP)
    recall = 0 if TP + FN == 0 else TP / (TP + FN)
    accurary = 0 if TP + TN + FN + FP == 0 else (TP + TN) / (TP + TN + FN + FP)
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    print(f1)


if __name__ == '__main__':
    cal_main()

import os
import json
import numpy as np


def load_data(path):
    transcripts = []
    scenarios = []
    with open(path, "r") as f:
        data = json.load(f)
    
    for elem in data:
        trn_1 = data[elem]["transcript_1"].rstrip()
        trn_2 = data[elem]["transcript_2"].rstrip()
        transcripts.append(trn_1 + " " + trn_2)
        scenarios.append(data[elem]["scenario_1"].rstrip())

    return transcripts, scenarios


def label_to_idx(labels):
    label2idx = {}
    for label in labels:
        if label not in label2idx:
            label2idx[label] = len(label2idx)

    return label2idx


def combine_data(trainscripts, labels):
    res = []
    for i in range(len(trainscripts)):
        res.append((trainscripts[i], labels[i]))

    return res

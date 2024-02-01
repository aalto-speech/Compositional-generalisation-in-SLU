import os
import json
import numpy as np


def load_data(path):
    transcripts = []
    scenarios = []
    with open(path, "r") as f:
        data = json.load(f)
    
    for elem in data:
        transcripts.append(data[elem]["transcript"].rstrip())
        scenarios.append(data[elem]["scenario"].rstrip())

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

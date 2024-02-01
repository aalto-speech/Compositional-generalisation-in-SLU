import os
import json
import numpy as np
import string

import torch


def load_data(path):
    transcripts = []
    labels = []
    with open(path, "r") as f:
        data = json.load(f)
    
    for elem in data:
        transcripts_temp = []
        labels_temp = []
        transcript = data[elem]["transcript"].rstrip()
        transcript = transcript.split(" ")
        for word in transcript:
            if "<" in word and ">" in word:
                labels_temp.append(word)
            else:
                #remove punctuation
                word = [c for c in word if c not in string.punctuation]
                word = "".join(word)
                word = word.replace("'", "")
                transcripts_temp.append(word)
        transcripts_temp = " ".join(transcripts_temp)
        transcripts.append(transcripts_temp)
        labels.append(labels_temp)


    return transcripts, labels


def label_to_idx(labels):
    label2idx = {}
    for sent in labels:
        for label in sent:
            if label not in label2idx:
                label2idx[label] = len(label2idx)

    # add CLS and SEP tokens
    label2idx["<CLS>"] = len(label2idx)
    label2idx["<SEP>"] = len(label2idx)
    return label2idx


def convert_label_to_idx(labels, label2idx):
    idx_labels = []
    for sent in labels:
        temp = []
        for label in sent:
            temp.append(label2idx[label])
        idx_labels.append(temp)

    return idx_labels


def map_tokenized_input_to_label_level(all_input_ids, all_labels, tokenizer, label2idx):
    labels_mapped = []
    for labels, input_ids in zip(all_labels, all_input_ids):
        labels_mapped_sent = []
        text_inputs = tokenizer.convert_ids_to_tokens(input_ids)
        # remove the special tokens
        text_inputs = [word for word in text_inputs if word != "[CLS]" and word != "[SEP]" and word != "[PAD]"]
        j = 0
        # add CLS token
        labels_mapped_sent.append(label2idx["<CLS>"])
        for i in range(len(text_inputs)):
            word = text_inputs[i]
            if "##" in word:
                labels_mapped_sent.append(labels[j - 1])
            else:
                # print(labels, j)
                labels_mapped_sent.append(labels[j])
                j += 1
        
        # add SEP token
        labels_mapped_sent.append(label2idx["<SEP>"])
        labels_mapped.append(labels_mapped_sent)
    
    max_len = max(len(label) for label in labels_mapped)
    padded_labels_mapped = [label + [0] * (max_len - len(label)) for label in labels_mapped]
    padded_labels_mapped = torch.tensor(padded_labels_mapped)

    return padded_labels_mapped

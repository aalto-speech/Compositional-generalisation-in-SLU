import os
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from transformers import BertTokenizer, BertForTokenClassification


import utils.prepare_data as prepare_data
import config
import train
from utils.evaluate_model import get_predictions


torch.manual_seed(2024)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Load the data
train_text, train_labels = prepare_data.load_data("/m/teamwork/t40511_asr/c/SLURP/speechbrain_data/entity_splits/difficult_splits/train.json")
dev_text, dev_labels = prepare_data.load_data("/m/teamwork/t40511_asr/c/SLURP/speechbrain_data/entity_splits/difficult_splits/dev.json")
test_text, test_labels = prepare_data.load_data("/m/teamwork/t40511_asr/c/SLURP/speechbrain_data/entity_splits/difficult_splits/test.json")

# Tokenize the text
train_tokenized = tokenizer(train_text, padding=True, truncation=True, max_length=128, return_tensors='pt')
dev_tokenized = tokenizer(dev_text, padding=True, truncation=True, max_length=128, return_tensors='pt')
test_tokenized = tokenizer(test_text, padding=True, truncation=True, max_length=128, return_tensors='pt')

# Extract the ids and attention mask
train_input_ids = train_tokenized["input_ids"]
train_attention_masks = train_tokenized["attention_mask"]
dev_input_ids = dev_tokenized["input_ids"]
dev_attention_masks = dev_tokenized["attention_mask"]
test_input_ids = test_tokenized["input_ids"]
test_attention_masks = test_tokenized["attention_mask"]

# Convert labels to indices
label2idx = prepare_data.label_to_idx(train_labels + dev_labels + test_labels)
idx2label = {v: k for k, v in label2idx.items()}
train_labels = prepare_data.convert_label_to_idx(train_labels, label2idx)
dev_labels = prepare_data.convert_label_to_idx(dev_labels, label2idx)
test_labels = prepare_data.convert_label_to_idx(test_labels, label2idx)

# Convert labels to tensors
# max_label_length_train = max(len(label) for label in train_labels)
# train_padded_labels = [label + [0] * (max_label_length_train - len(label)) for label in train_labels]
# train_padded_labels = torch.tensor(train_padded_labels)

# max_label_length_dev = max(len(label) for label in dev_labels)
# dev_padded_labels = [label + [0] * (max_label_length_dev - len(label)) for label in dev_labels]
# dev_padded_labels = torch.tensor(dev_padded_labels)

# max_label_length_test = max(len(label) for label in test_labels)
# test_padded_labels = [label + [0] * (max_label_length_test - len(label)) for label in test_labels]
# test_padded_labels = torch.tensor(test_padded_labels)

# Map tokenized input to label-level granularity
padded_train_labels = prepare_data.map_tokenized_input_to_label_level(train_input_ids, train_labels, tokenizer, label2idx)
padded_dev_labels = prepare_data.map_tokenized_input_to_label_level(dev_input_ids, dev_labels, tokenizer, label2idx)
padded_test_labels = prepare_data.map_tokenized_input_to_label_level(test_input_ids, test_labels, tokenizer, label2idx)

# print(len(tokenizer.convert_ids_to_tokens(train_input_ids[0])), len(padded_train_labels[0]))
# print(tokenizer.convert_ids_to_tokens(train_input_ids[0]), padded_train_labels[0])

# Create datasets
train_dataset = TensorDataset(train_input_ids, train_attention_masks, padded_train_labels)
dev_dataset = TensorDataset(dev_input_ids, dev_attention_masks, padded_dev_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, padded_test_labels)

# Create a dataloader
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load the BERT model
bert_model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(label2idx))
bert_model.to(device)

# Create output directory if it doesn't exist
if not os.path.exists(config.save_dir):
    os.makedirs(config.save_dir)


if config.skip_training == False:
    print('Training...')
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    optimizer = torch.optim.AdamW(bert_model.parameters(), lr=config.lr) 


    train.trainer(
        model=bert_model,
        criterion=criterion,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        dev_dataloader=dev_dataloader,
        n_epochs=config.n_epochs,
        idx2label=idx2label,
        save_dir=config.save_dir,
        device=device
    )
else:
    print("Loading the model...")
    saved_models = []
    for file in os.listdir(config.save_dir):
        if file.endswith(".pt"):
            saved_models.append(file)
    # order the models by epoch
    saved_models.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))
    best_model = saved_models[0]
    print("Loading model: ", best_model)

    checkpoint = torch.load(os.path.join(config.save_dir, best_model), map_location=torch.device("cpu"))
    bert_model.load_state_dict(checkpoint["model"])
    bert_model.to(device)
    bert_model.eval()

# Test the model
get_predictions(bert_model, test_dataloader, tokenizer, label2idx, device)

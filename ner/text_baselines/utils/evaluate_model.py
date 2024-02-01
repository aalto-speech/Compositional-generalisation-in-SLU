from sklearn.metrics import f1_score
import numpy as np


def get_predictions(model, test_dataloader, tokenizer, label2idx, device):
    true_labels = []
    pred_labels = []
    texts = []
    lengths = []
    for batch in test_dataloader:
        input_ids, attention_mask, labels = batch

        # convert indices to text
        text = tokenizer.convert_ids_to_tokens(input_ids[0])
        texts.append(text)
        length = 0
        for i, elem in enumerate(text):
            if elem != "[PAD]":
                length += 1
        lengths.append(length)

        # Move to GPU
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        topi, topk = outputs.logits.topk(1)
        true_labels.extend(labels.tolist())
        pred_labels.extend(topk.tolist())


    # Prepare the true labels
    true_labels_clean = []
    for i in range(len(true_labels)):
        true_tag_sent = true_labels[i]
        true_tag_sent = true_tag_sent[:lengths[i]]
        # remove the CLS and SEP tokens
        true_tag_sent = [tag for tag in true_tag_sent if tag != label2idx["<CLS>"] and tag != label2idx["<SEP>"]]
        for tag in true_tag_sent:
            true_labels_clean.append(tag)

    # Prepare the predicted labels
    pred_labels_clean = []
    for i in range(len(pred_labels)):
        pred_tag_sent = pred_labels[i]
        pred_tag_sent = pred_tag_sent[:lengths[i]]
        # remove the CLS and SEP tokens
        pred_tag_sent = [tag[0] for tag in pred_tag_sent if tag[0] != label2idx["<CLS>"] and tag[0] != label2idx["<SEP>"]]
        for tag in pred_tag_sent:
            pred_labels_clean.append(tag)

    true_labels_clean = np.array(true_labels_clean)
    pred_labels_clean = np.array(pred_labels_clean)
    # # flatten the arrays
    # true_labels_clean = true_labels_clean.flatten()
    # pred_labels_clean = pred_labels_clean.flatten()

    # Calculate F1 score
    micro_f1 = f1_score(true_labels_clean, pred_labels_clean, average="micro")

    print("Micro F1 score: {:.2f}".format(micro_f1 * 100))

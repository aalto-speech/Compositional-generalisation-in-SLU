from sklearn.metrics import f1_score


def get_predictions(model, test_dataloader, device):
    true_labels = []
    pred_labels = []
    for batch in test_dataloader:
        input_ids, attention_mask, labels = batch

        # Move to GPU
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        topi, topk = outputs.logits.topk(1)
        true_labels.extend(labels.tolist())
        pred_labels.extend(topk.squeeze(1).tolist())

    # Calculate F1 score
    micro_f1 = f1_score(true_labels, pred_labels, average="micro")

    print("Micro F1 score: {:.2f}".format(micro_f1 * 100))

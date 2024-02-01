import os
from sklearn.metrics import f1_score
import numpy as np
import torch


def trainer(model, criterion, optimizer, train_dataloader, dev_dataloader, n_epochs, idx2label, save_dir, device):
    initial_micro_f1 = 0.0
    for epoch in range(n_epochs):
        model.train()
        
        avg_loss_train = 0.0
        train_total_loss = 0.0

        avg_loss_dev = 0.0
        dev_total_loss = 0.0

        for batch in train_dataloader:
            input_ids, attention_mask, labels = batch

            # Move to GPU
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            train_loss = outputs.loss

            # Backward pass
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_total_loss += train_loss.detach().item()
        avg_loss_train = train_total_loss / len(train_dataloader)

        # VALIDATION
        with torch.no_grad():
            model.eval()
            true_labels = []
            pred_labels = []
            for batch in dev_dataloader:
                input_ids, attention_mask, labels = batch

                # Move to GPU
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                dev_loss = outputs.loss

                topi, topk = outputs.logits.topk(1)
                true_labels.extend(labels.tolist())
                pred_labels.extend(topk.squeeze(1).tolist())

                dev_total_loss += dev_loss.detach().item()
            avg_loss_dev = dev_total_loss / len(dev_dataloader)
        

        # Calculate F1 score
        # flatten the arrays
        true_labels = np.array(true_labels)
        true_labels = true_labels.flatten()

        pred_labels_flat = []
        for sent in pred_labels:
            for label in sent:
                pred_labels_flat.append(label[0])

        pred_labels_flat = np.array(pred_labels_flat)

        
        #pred_labels_flat = pred_labels_flat.flatten()
        micro_f1 = f1_score(true_labels, pred_labels_flat, average="micro")

        print("[Epoch: %d] train_loss: %.4f    val_loss: %.4f   f1: %.4f" % (epoch+1, avg_loss_train, avg_loss_dev, micro_f1))

        # save the losses to a file
        with open(os.path.join(save_dir, "log.txt"), "a") as f:
            f.write("[Epoch: %d] train_loss: %.4f    val_loss: %.4f   f1: %.4f \n " % (epoch+1, avg_loss_train, avg_loss_dev, micro_f1))

        if micro_f1 > initial_micro_f1:
            # delete all previous models, except the last epoch model, which we can use to continue training
            for file in os.listdir(save_dir):
                if file.endswith(".pt") and file != "state_dict_" + str(n_epochs) + ".pt":
                    os.remove(os.path.join(save_dir, file))

            print("saving the model...")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                }, os.path.join(save_dir, "state_dict_" + str(epoch+1) + ".pt"))
            print("Done...")
            initial_micro_f1 = micro_f1

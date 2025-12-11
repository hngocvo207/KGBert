import argparse
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import BertTokenizer, BertModel
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


class DevDataset(Dataset):
    def __init__(self, tsv_file, tokenizer, max_len=64):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(tsv_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if line_idx == 0:
                    continue
                cols = line.strip().split('\t')
                if len(cols) < 3:
                    continue
                label = int(cols[1])
                text = cols[2]
                self.samples.append((label, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, text = self.samples[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BertClassifier(nn.Module):
    def __init__(self, pretrained_model_path, num_labels=2, freeze_bert=False):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits, pooled_output


def compute_metrics(labels, preds, probs):
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    try:
        auc_roc = roc_auc_score(labels, probs)
    except ValueError:
        auc_roc = 0.0

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0.0

    return {
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUC-ROC': auc_roc,
        'FNR': fnr,
        'FPR': fpr
    }


def plot_training_curve(train_losses, val_losses, save_path):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, 'bo-', label='Train Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Training curve saved to: {save_path}")


def direct_evaluate(model, dataloader, device, run_dir):
    model.eval()
    preds_list = []
    labels_list = []
    probs_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating (Direct)"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].cpu().numpy()

            logits, _ = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            predicted = torch.argmax(logits, dim=1).cpu().numpy()

            preds_list.extend(predicted)
            labels_list.extend(labels)
            probs_list.extend(probs)

    metrics_result = compute_metrics(labels_list, preds_list, probs_list)
    result_path = os.path.join(run_dir, "direct_evaluate_results.txt")
    with open(result_path, "w") as f:
        f.write(str(metrics_result))

    print(f"[Direct Evaluate] Results saved: {metrics_result}")
    print(f"File saved to: {result_path}")
    return metrics_result


def finetune(model, train_loader, val_loader, device, epochs, lr, run_dir):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    model.to(device)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_model_path = os.path.join(run_dir, "best_ft_model.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Fine-tuning Epoch {epoch}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_epoch_loss = running_loss / len(train_loader)
        train_losses.append(train_epoch_loss)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                logits, _ = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()

        val_epoch_loss = val_loss / len(val_loader)
        val_losses.append(val_epoch_loss)

        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model, best_model_path)
            print(f"Best model saved with val loss: {best_val_loss:.4f} in {best_model_path}")

    model_save_path = os.path.join(run_dir, "finetuned_model.pth")
    torch.save(model, model_save_path)
    print(f"Finetuned model saved: {model_save_path}")


def linear_probe(model, train_loader, val_loader, device, epochs, lr, run_dir):
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    model.to(device)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_model_path = os.path.join(run_dir, "best_linear_probe_model.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Linear Probe Epoch {epoch}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_epoch_loss = running_loss / len(train_loader)
        train_losses.append(train_epoch_loss)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                logits, _ = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()

        val_epoch_loss = val_loss / len(val_loader)
        val_losses.append(val_epoch_loss)

        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model, best_model_path)
            print(f"Best model saved with val loss: {best_val_loss:.4f} in {best_model_path}")

    model_save_path = os.path.join(run_dir, "linear_probe_model.pth")
    torch.save(model, model_save_path)
    print(f"Linear probe model saved: {model_save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="linear",
                        choices=["direct", "finetune", "linear", "aft_ft"])
    parser.add_argument("--dev_tsv", type=str, default="Data/SPN/dev.tsv", help="Path to the dev dataset file")
    parser.add_argument("--pretrained_path", type=str, default="Train_output/01_09_17_47")
    parser.add_argument("--model_path", type=str, default="Eval_output/finetune/01_09_22_50/best_ft_model.pth")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--max_length", type=int, default=256, help="Max length for BERT input")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")
    run_dir = f"Eval_output/{args.task}/{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    print(f"Running task: {args.task}")
    config_txt_path = os.path.join(run_dir, "config.txt")
    with open(config_txt_path, "w") as f:
        f.write("Running Configurations:\n")
        f.write(str(args) + "\n")

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    with open(config_txt_path, "a") as f:
        f.write(f"Device: {device}\n")

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_path)

    dataset = DevDataset(tsv_file=args.dev_tsv, tokenizer=tokenizer, max_len=args.max_length)
    print(f"Loaded dataset from {args.dev_tsv}, total samples: {len(dataset)}")
    with open(config_txt_path, "a") as f:
        f.write(f"Loaded dataset from {args.dev_tsv}, total samples: {len(dataset)}\n")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    freeze_bert_flag = True if args.task == "linear" else False
    model = BertClassifier(
        pretrained_model_path=args.pretrained_path,
        num_labels=2,
        freeze_bert=freeze_bert_flag
    )
    print(f"Model loaded, freeze_bert={freeze_bert_flag}")
    with open(config_txt_path, "a") as f:
        f.write(f"Model loaded. freeze_bert={freeze_bert_flag}\n")

    if args.task == "direct":
        model.eval()
        model.to(device)
        direct_metrics = direct_evaluate(model, val_loader, device, run_dir)
        print(f"Direct evaluate results: {direct_metrics}")

        result_txt_path = os.path.join(run_dir, "result.txt")
        with open(result_txt_path, "w") as f:
            f.write("Direct Evaluate Results:\n")
            f.write(str(direct_metrics) + "\n")

    elif args.task == "aft_ft":
        model = torch.load(args.model_path)
        model.eval()
        model.to(device)
        direct_metrics = direct_evaluate(model, val_loader, device, run_dir)
        print(f"Evaluate results after supervised training: {direct_metrics}")

        result_txt_path = os.path.join(run_dir, "result.txt")
        with open(result_txt_path, "w") as f:
            f.write("Evaluate Results After Supervised Training:\n")
            f.write(str(direct_metrics) + "\n")

    elif args.task == "finetune":
        finetune(model, train_loader, val_loader, device,
                 epochs=args.epochs,
                 lr=args.learning_rate,
                 run_dir=run_dir)

        model.eval()
        model.to(device)
        preds_list = []
        labels_list = []
        probs_list = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].cpu().numpy()

                logits, _ = model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                predicted = torch.argmax(logits, dim=1).cpu().numpy()
                preds_list.extend(predicted)
                labels_list.extend(labels)
                probs_list.extend(probs)

        metrics_result = compute_metrics(labels_list, preds_list, probs_list)
        print(f"[Finetune Evaluate] {metrics_result}")

        result_txt_path = os.path.join(run_dir, "result.txt")
        with open(result_txt_path, "w") as f:
            f.write("Finetune Evaluate Results:\n")
            f.write(str(metrics_result) + "\n")

    elif args.task == "linear":
        linear_probe(model, train_loader, val_loader, device,
                     epochs=args.epochs,
                     lr=1e-3,
                     run_dir=run_dir)

        model.eval()
        model.to(device)
        preds_list = []
        labels_list = []
        probs_list = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].cpu().numpy()

                logits, _ = model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                predicted = torch.argmax(logits, dim=1).cpu().numpy()
                preds_list.extend(predicted)
                labels_list.extend(labels)
                probs_list.extend(probs)

        metrics_result = compute_metrics(labels_list, preds_list, probs_list)
        print(f"[Linear Probe Evaluate] {metrics_result}")

        result_txt_path = os.path.join(run_dir, "result.txt")
        with open(result_txt_path, "w") as f:
            f.write("Linear Probe Evaluate Results:\n")
            f.write(str(metrics_result) + "\n")


if __name__ == "__main__":
    main()
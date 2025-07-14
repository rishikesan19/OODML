import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import sys
import torch.nn.functional as F
import wandb

# Add the parent directory to the path to find other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from oodml.models import OODML
from oodml.utils import make_parse, EarlyStopping_OODML, calculate_metrics
from datasets.load_datasets import h5file_Dataset

def train_one_epoch(model, train_loader, optimizer, device, args, epoch):
    model.train()
    total_loss = 0
    all_labels = []
    all_probs = []

    for i, (coords, data, label) in enumerate(train_loader):
        data, label = data.squeeze(0).to(device), label.to(device).long()
        coords = coords.squeeze(0).to(device)
        data = data.float()
        
        optimizer.zero_grad()
        
        results_dict = model(data, coords)

        Y_hat_ER = results_dict['Y_hat_ER']
        Y_hat_IR = results_dict['Y_hat_IR']
        Y_hat_DM = results_dict['Y_hat_DM']
        pseudo_label_preds = results_dict['pseudo_label_preds']

        loss_er = F.cross_entropy(Y_hat_ER, label)
        loss_ir = F.cross_entropy(Y_hat_IR, label)
        loss_dm = F.cross_entropy(Y_hat_DM, label)

        A_ER = results_dict['A_ER']
        A_IR = results_dict['A_IR']
        
        with torch.no_grad():
            # FIX: Use squeeze(1) on A_IR to correctly reduce dimensions from [1, 1, T] to [1, T]
            y_pseudo = torch.sigmoid((A_ER * Y_hat_ER.softmax(1)[:,1].unsqueeze(1) + A_IR.squeeze(1) * Y_hat_IR.softmax(1)[:,1].unsqueeze(1)) / args.tau)

        if pseudo_label_preds.shape != y_pseudo.shape:
             y_pseudo = y_pseudo.permute(1, 0)
        
        loss_plce = F.binary_cross_entropy(pseudo_label_preds.squeeze(), y_pseudo.squeeze())

        total_epoch_loss = loss_er + loss_ir + args.lambda_plce * loss_plce + loss_dm

        total_epoch_loss.backward()
        optimizer.step()
        
        total_loss += total_epoch_loss.item()
        
        all_labels.append(label.cpu().numpy())
        all_probs.append(F.softmax(Y_hat_DM, dim=1).detach().cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    acc, auc, f1, precision, recall = calculate_metrics(all_labels, all_probs)

    print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Acc: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
    
    if args.wandb:
        wandb.log({
            "epoch": epoch + 1, "train/loss": avg_loss, "train/accuracy": acc, "train/auc": auc,
            "train/f1_score": f1, "train/precision": precision, "train/recall": recall
        })

def validate(model, val_loader, device, args, epoch):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for i, (coords, data, label) in enumerate(val_loader):
            data, label = data.squeeze(0).to(device), label.to(device).long()
            coords = coords.squeeze(0).to(device)
            data = data.float()
            
            results_dict = model(data, coords)
            Y_hat_DM = results_dict['Y_hat_DM']
            
            probs = F.softmax(Y_hat_DM, dim=1)
            
            all_labels.append(label.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    acc, auc, f1, precision, recall = calculate_metrics(all_labels, all_probs)
    print(f"Epoch {epoch+1} - Val Acc: {acc:.4f}, Val AUC: {auc:.4f}, Val F1: {f1:.4f}")

    if args.wandb:
        wandb.log({
            "epoch": epoch + 1, "val/accuracy": acc, "val/auc": auc,
            "val/f1_score": f1, "val/precision": precision, "val/recall": recall
        })
    
    return auc

def main():
    args = make_parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.wandb:
        wandb.init(project="OODML-WSI-Classification", name=args.exp_name, config=args)
    
    full_train_dataset = h5file_Dataset(args.csv, args.h5_path, 'train')
    
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = OODML(input_dim=args.input_dim, n_classes=args.n_classes, K=args.K).to(device)
    optimizer = optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.wandb:
        wandb.watch(model, log="all")
    
    early_stopping = EarlyStopping_OODML(patience=20, verbose=True)

    for epoch in range(args.epochs):
        print(f"--- Epoch {epoch+1}/{args.epochs} ---")
        train_one_epoch(model, train_loader, optimizer, device, args, epoch)
        val_auc = validate(model, val_loader, device, args, epoch)
        
        early_stopping(val_auc, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
            
    print("\nTraining finished.")
    if args.wandb:
        wandb.finish()

if __name__ == '__main__':
    main()

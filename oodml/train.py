import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import sys
import torch.nn.functional as F
import math

# Add the parent directory to the path to find other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from oodml.models import OODML
from oodml.utils import make_parse, ModelCheckpoint, calculate_metrics
from datasets.load_datasets import h5file_Dataset

def get_lambda_plce(epoch, max_epochs, max_lambda=0.5):
    """Calculates lambda_plce with a cosine annealing schedule, as per paper."""
    return max_lambda * 0.5 * (1 + math.cos(math.pi * epoch / max_epochs))

def train_one_epoch(model, train_loader, optimizer, device, args, epoch):
    model.train()
    total_loss = 0
    all_labels, all_probs = [], []

    for i, (coords, data, label) in enumerate(train_loader):
        data, label = data.squeeze(0).to(device), label.to(device).long()
        coords = coords.squeeze(0).to(device)
        data = data.float()
        
        optimizer.zero_grad()
        results_dict = model(data, coords)

        Y_hat_ER, Y_hat_IR, Y_hat_DM = results_dict['Y_hat_ER'], results_dict['Y_hat_IR'], results_dict['Y_hat_DM']
        
        loss_er = F.cross_entropy(Y_hat_ER, label)
        loss_ir = F.cross_entropy(Y_hat_IR, label)
        loss_dm = F.cross_entropy(Y_hat_DM, label)
        
        lambda_plce = get_lambda_plce(epoch, args.epochs, args.lambda_plce)
        pseudo_label_preds = results_dict['pseudo_label_preds']
        
        if pseudo_label_preds.numel() > 0:
            A_ER, A_IR = results_dict['A_ER'], results_dict['A_IR']
            with torch.no_grad():
                y_pseudo = torch.sigmoid((A_ER * Y_hat_ER.softmax(1)[:,1].unsqueeze(1) + A_IR.squeeze(1) * Y_hat_IR.softmax(1)[:,1].unsqueeze(1)) / args.tau)
            if pseudo_label_preds.shape != y_pseudo.shape:
                 y_pseudo = y_pseudo.permute(1, 0)
            loss_plce = F.binary_cross_entropy(pseudo_label_preds.squeeze(), y_pseudo.squeeze())
        else:
            loss_plce = 0.0
            
        total_epoch_loss = loss_er + loss_ir + loss_dm + (lambda_plce * loss_plce)

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

def validate(model, val_loader, device, args, epoch):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for i, (coords, data, label) in enumerate(val_loader):
            data, label = data.squeeze(0).to(device), label.to(device).long()
            coords = coords.squeeze(0).to(device)
            data = data.float()
            results_dict = model(data, coords)
            probs = F.softmax(results_dict['Y_hat_DM'], dim=1)
            all_labels.append(label.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    acc, auc, f1, precision, recall = calculate_metrics(all_labels, all_probs)
    print(f"Epoch {epoch+1} - Val Acc: {acc:.4f}, Val AUC: {auc:.4f}, Val F1: {f1:.4f}")
    
    # --- Return both metrics ---
    return f1, auc

def main():
    args = make_parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ... (Data loading and model initialization) ...
    full_train_dataset = h5file_Dataset(args.csv, args.h5_path, 'train')
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    model = OODML(input_dim=args.input_dim, n_classes=args.n_classes, K=args.K, pseudo_bag_size=args.pseudo_bag_size).to(device)
    optimizer = optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # --- Create two checkpoint savers ---
    f1_checkpoint = ModelCheckpoint(patience=50, verbose=True, path=args.best_f1_ckpt_path, metric_name='F1 Score')
    auc_checkpoint = ModelCheckpoint(patience=50, verbose=True, path=args.best_auc_ckpt_path, metric_name='AUC')

    # --- Training Loop ---
    for epoch in range(args.epochs):
        print(f"--- Epoch {epoch+1}/{args.epochs} ---")
        train_one_epoch(model, train_loader, optimizer, device, args, epoch)
        val_f1, val_auc = validate(model, val_loader, device, args, epoch)
        
        # --- Update both checkpoint savers ---
        f1_checkpoint(val_f1, model)
        auc_checkpoint(val_auc, model)
        
        # Stop training if BOTH metrics have stopped improving
        if f1_checkpoint.early_stop and auc_checkpoint.early_stop:
            print("Early stopping as both F1 and AUC have stopped improving.")
            break
            
    print("\nTraining finished.")
    
    # Save the final model state
    print(f"Saving final model to {args.last_ckpt_path}")
    torch.save(model.state_dict(), args.last_ckpt_path)

if __name__ == '__main__':
    main()
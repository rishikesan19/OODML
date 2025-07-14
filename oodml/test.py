import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import torch.nn.functional as F

# Add the parent directory to the path to find other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from oodml.models import OODML
from oodml.utils import make_parse, calculate_metrics
from datasets.load_datasets import h5file_Dataset

def test_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for i, (coords, data, label) in enumerate(test_loader):
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
    
    print("\n--- Final Test Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("--------------------------")

def main():
    args = make_parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- DATA LOADING ---
    test_dataset = h5file_Dataset(args.csv, args.h5_path, 'test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --- MODEL LOADING ---
    print(f"Loading model from checkpoint: {args.ckpt_path}")
    model = OODML(input_dim=args.input_dim, n_classes=args.n_classes, K=args.K).to(device)
    
    try:
        model.load_state_dict(torch.load(args.ckpt_path))
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.ckpt_path}")
        sys.exit(1)
        
    # --- RUN TESTING ---
    test_model(model, test_loader, device)

if __name__ == '__main__':
    main()

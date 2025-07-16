import argparse
import torch
import os
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

def make_parse():
    parser = argparse.ArgumentParser(description='OODML Training & Testing - Paper Implementation')
    
    # --- Settings from the paper ---
    parser.add_argument('--exp_name', default='oodml_paper_impl', type=str, help='Name of the experiment')
    parser.add_argument('--epochs', default=200, type=int, help='Number of total epochs to run')
    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay for AdaMax optimizer')

    # --- Data settings ---
    parser.add_argument('--csv', default='./camelyon16.csv', type=str, help='Path to the CSV file')
    parser.add_argument('--h5_path', default='./patch_feats_pretrain_natural_supervised.h5', type=str, help='Path to the H5 feature file')
    
    # --- Model hyperparameters ---
    parser.add_argument('--input_dim', default=1024, type=int, help='Dimension of instance features')
    parser.add_argument('--n_classes', default=2, type=int, help='Number of classes')
    parser.add_argument('--K', default=5, type=int, help='Size of the Adaptive Memory Bank (AMB)')
    parser.add_argument('--embed_dim', default=512, type=int, help='Internal embedding dimension')
    parser.add_argument('--pseudo_bag_size', default=512, type=int, help='Number of instances per pseudo-bag')
    parser.add_argument('--tau', default=1.0, type=float, help='Temperature for pseudo-label generation')
    parser.add_argument('--lambda_plce', default=0.5, type=float, help='Max weight for pseudo-label loss')

    # --- Checkpoint Paths ---
    parser.add_argument('--best_f1_ckpt_path', default='checkpoints/oodml_best_f1_model.pt', type=str, help='Path for saving the best F1 model')
    parser.add_argument('--best_auc_ckpt_path', default='checkpoints/oodml_best_auc_model.pt', type=str, help='Path for saving the best AUC model')
    parser.add_argument('--last_ckpt_path', default='checkpoints/oodml_last_model.pt', type=str, help='Path for saving the final model')

    args = parser.parse_args()
    return args

def calculate_metrics(labels, probs):
    predictions = np.argmax(probs, axis=1)
    acc = accuracy_score(labels, predictions)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = 0.5
    f1 = f1_score(labels, predictions, zero_division=0)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    return acc, auc, f1, precision, recall

class ModelCheckpoint:
    def __init__(self, patience=25, verbose=True, delta=0, path='checkpoints/best_model.pt', metric_name='F1 Score'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_val_score = 0
        self.delta = delta
        self.path = path
        self.metric_name = metric_name

    def __call__(self, val_score, model):
        score = val_score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter for {self.metric_name}: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_score, model):
        if self.verbose:
            print(f'Validation {self.metric_name} improved ({self.best_val_score:.4f} --> {val_score:.4f}). Saving model to {self.path}')
        dir_name = os.path.dirname(self.path)
        os.makedirs(dir_name, exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.best_val_score = val_score
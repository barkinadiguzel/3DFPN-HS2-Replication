import torch

def sensitivity(preds, targets):
    tp = ((preds>0.5) & (targets==1)).sum().float()
    fn = ((preds<=0.5) & (targets==1)).sum().float()
    return tp / (tp + fn + 1e-6)

def specificity(preds, targets):
    tn = ((preds<=0.5) & (targets==0)).sum().float()
    fp = ((preds>0.5) & (targets==0)).sum().float()
    return tn / (tn + fp + 1e-6)

def false_positives_per_scan(preds, targets):
    fp = ((preds>0.5) & (targets==0)).sum().float()
    n_scans = targets.shape[0]
    return fp / n_scans

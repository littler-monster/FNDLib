import os
import random
import numpy as np
import torch
from sklearn import metrics


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metric_new(pred, pred_old, label,mode):
    p = metrics.precision_score(label, pred, zero_division=0,average='weighted')
    r = metrics.recall_score(label, pred,average='weighted')
    F1 = metrics.f1_score(label, pred,average='weighted')
    acc = metrics.accuracy_score(label, pred)
    fpr, tpr, thresholds = metrics.roc_curve(label, pred_old.detach().numpy()[:,1], pos_label=1)
    auc = metrics.auc(fpr, tpr)

    if mode == 'test':
        print(metrics.classification_report(label, pred, digits=4))
    return p, r, F1, acc, auc
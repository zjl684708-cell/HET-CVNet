import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from easydict import EasyDict
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def cont_grad(x, rate=1):
    return rate * x + (1 - rate) * x.detach()

def find_best_threshold(y_trues, y_preds):
    '''
        This function is utilized to find the threshold corresponding to the best ACER
        Args:
            y_trues (list): the list of the ground-truth labels, which contains the int data
            y_preds (list): the list of the predicted results, which contains the float data
    '''
    print("Finding best threshold...")
    best_thre = 0.5
    best_metrics = None
    candidate_thres = list(np.unique(np.sort(y_preds)))
    for thre in candidate_thres:
        metrics = cal_metrics(y_trues, y_preds, threshold=thre)
        if best_metrics is None:
            best_metrics = metrics
            best_thre = thre
        elif metrics.ACER < best_metrics.ACER:
            best_metrics = metrics
            best_thre = thre
    print(f"Best threshold is {best_thre}")
    return best_thre, best_metrics


def cal_metrics(y_trues, y_preds, threshold=0.5):
    '''
        Calculate performance metrics.
        Args:
            y_trues (list/np.ndarray): ground-truth labels (0/1)
            y_preds (list/np.ndarray): predicted scores (0~1 float)
            threshold:
                'best' -> use the threshold that minimizes ACER
                'auto' -> use EER threshold
                float  -> use given threshold
    '''
    metrics = EasyDict()

    # ROC / AUC / EER & its threshold
    fpr, tpr, thresholds = roc_curve(y_trues, y_preds)
    metrics.AUC = auc(fpr, tpr)

    # EER (solve 1 - x - TPR(FPR=x) = 0)
    metrics.EER = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    metrics.Thre = float(interp1d(fpr, thresholds)(metrics.EER))

    if threshold == 'best':
        _, best_metrics = find_best_threshold(y_trues, y_preds)
        return best_metrics
    elif threshold == 'auto':
        threshold = metrics.Thre  # EER threshold

    # Binarize predictions
    prediction = (np.array(y_preds) > float(threshold)).astype(int)

    # Confusion matrix with labels=[0,1] returns:
    # [[TN, FP],
    #  [FN, TP]]
    cm = confusion_matrix(y_trues, prediction, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    # Accuracy / Precision / Recall（含除零保护）
    total = TN + FP + FN + TP
    metrics.ACC = (TP + TN) / total if total > 0 else 0.0  # 保留原键
    metrics.Accuracy = metrics.ACC                           # 新增：Accuracy（同义）

    metrics.Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    metrics.Recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # APCER / BPCER / ACER（按标准定义）
    metrics.APCER = FP / (TN + FP) if (TN + FP) > 0 else 0.0   # False Acceptance Rate
    metrics.BPCER = FN / (FN + TP) if (FN + TP) > 0 else 0.0   # False Rejection Rate
    metrics.ACER  = (metrics.APCER + metrics.BPCER) / 2

    return metrics



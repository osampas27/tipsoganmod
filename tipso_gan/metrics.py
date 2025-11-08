import json, time
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn=fp=fn=tp=0
    return {'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec), 'f1': float(f1), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp), 'tn': int(tn)}, cm.tolist()
def save_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
class Timer:
    def __enter__(self): self.t0=time.perf_counter(); return self
    def __exit__(self,*a): self.elapsed=time.perf_counter()-self.t0

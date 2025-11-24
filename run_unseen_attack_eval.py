import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import numpy as np
from sklearn.model_selection import train_test_split
from tipso_gan.train import TIPSOTrainer, cfg
from tipso_gan.cicids_loader import load_with_attack_names
from tipso_gan.metrics import compute_metrics, save_json
def main():
    os.makedirs('artifacts', exist_ok=True)
    X, y, names, feats = load_with_attack_names(['cicids2018.csv'])
    y = y.reshape(-1)
    uniq = np.unique(names)
    attacks = [n for n in uniq if str(n).upper() != 'BENIGN']
    if len(attacks) == 0:
        print('No attack families found; cannot run unseen eval.'); return
    counts = {a: int(np.sum(np.array(names) == a)) for a in attacks}
    held_out = max(counts.items(), key=lambda kv: kv[1])[0]
    mask_train = np.array([str(n) != held_out for n in names])
    mask_held  = np.array([str(n) == held_out for n in names])
    X_all_train, y_all_train = X[mask_train], y[mask_train]
    X_attack, y_attack       = X[mask_held],  y[mask_held]
    benign_idx = np.where(y == 0)[0]
    if len(X_attack) == 0 or len(benign_idx) == 0:
        print('Insufficient held-out attack or benign; skipping.'); return
    k = min(len(X_attack), len(benign_idx))
    Xte = np.vstack([X_attack[:k], X[benign_idx[:k]]])
    yte = np.hstack([y_attack[:k], y[benign_idx[:k]]])
    if len(X_all_train) < 20 or len(np.unique(y_all_train)) < 2:
        print('Training set too small after hold-out; skipping.'); return
    Xtr, Xv, ytr, yv = train_test_split(X_all_train, y_all_train, test_size=0.2, random_state=42, stratify=y_all_train)
    mask_normal = (ytr == 0)
    Xn = Xtr[mask_normal] if mask_normal.sum() >= 16 else Xtr[:max(16, len(Xtr)//4)]
    t = TIPSOTrainer(input_dim=X.shape[1])
    t.pretrain_psogan(Xn, epochs=cfg.epochs_pretrain, batch_size=cfg.batch_size)
    t.train_tipso(Xn, Xtr, ytr.reshape(-1,1), Xv, yv.reshape(-1,1), epochs=cfg.epochs_tipso, batch_size=cfg.batch_size, balance_strategy='class_weight')
    preds = t.dee.predict(Xte, verbose=0).argmax(axis=1)
    m, _ = compute_metrics(yte, preds)
    save_json('artifacts/unseen_report.json', {'held_out_attack': str(held_out), 'held_out_count': int(len(X_attack)), 'metrics': m})
    print('Held-out', held_out, '-> artifacts/unseen_report.json')
if __name__=='__main__': main()

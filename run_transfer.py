import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from tipso_gan.train import TIPSOTrainer, cfg
from tipso_gan.cicids_loader import load_cicids_csv_preset
from tipso_gan.metrics import compute_metrics, save_json
def split_domain(X, y, fraction=0.5):
    n = X.shape[0]; k = int(n*fraction)
    return (X[:k], y[:k]), (X[k:], y[k:])
def main():
    os.makedirs('artifacts', exist_ok=True)
    Xtr,ytr,Xv,yv,Xte,yte,feats = load_cicids_csv_preset(['sample_cicids_small.csv'])
    t_base = TIPSOTrainer(input_dim=Xtr.shape[1])
    (Xa, ya), (Xb, yb) = split_domain(Xtr, ytr, 0.5)
    t_base.pretrain_psogan(Xb[yb.flatten()==0], epochs=cfg.epochs_pretrain, batch_size=cfg.batch_size)
    t_base.train_tipso(Xb[yb.flatten()==0], Xb, yb, Xv, yv, epochs=cfg.epochs_tipso, batch_size=cfg.batch_size, balance_strategy='class_weight')
    preds_base = t_base.dee.predict(Xte, verbose=0).argmax(axis=1)
    base_metrics, _ = compute_metrics(yte, preds_base)
    t_tr = TIPSOTrainer(input_dim=Xtr.shape[1])
    t_tr.dee.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    t_tr.dee.fit(Xa, ya, epochs=5, batch_size=cfg.batch_size, verbose=0)
    t_tr.pretrain_psogan(Xb[yb.flatten()==0], epochs=cfg.epochs_pretrain, batch_size=cfg.batch_size)
    t_tr.train_tipso(Xb[yb.flatten()==0], Xb, yb, Xv, yv, epochs=cfg.epochs_tipso, batch_size=cfg.batch_size, balance_strategy='class_weight')
    preds_tr = t_tr.dee.predict(Xte, verbose=0).argmax(axis=1)
    tr_metrics, _ = compute_metrics(yte, preds_tr)
    save_json('artifacts/dee_transfer_report.json', {'baseline': base_metrics, 'transfer': tr_metrics})
    print('Wrote artifacts/dee_transfer_report.json')
if __name__=='__main__': main()

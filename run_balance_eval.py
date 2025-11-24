import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import csv
from tipso_gan.train import TIPSOTrainer, cfg
from tipso_gan.cicids_loader import load_cicids_csv_preset
from tipso_gan.metrics import compute_metrics
def main():
    os.makedirs('artifacts', exist_ok=True)
    Xtr,ytr,Xv,yv,Xte,yte,feats = load_cicids_csv_preset(['cicids2018.csv'])
    strategies = ['none','undersample','oversample','class_weight']
    rows = []
    for s in strategies:
        if (ytr.flatten()==0).sum() == 0:
            print(f"Skipping {s}: no normal in train."); continue
        t = TIPSOTrainer(input_dim=Xtr.shape[1])
        t.pretrain_psogan(Xtr[ytr.flatten()==0], epochs=cfg.epochs_pretrain, batch_size=cfg.batch_size)
        t.train_tipso(Xtr[ytr.flatten()==0], Xtr, ytr, Xv, yv, epochs=cfg.epochs_tipso, batch_size=cfg.batch_size, balance_strategy=s)
        preds = t.dee.predict(Xte, verbose=0).argmax(axis=1)
        m, _ = compute_metrics(yte, preds); m['strategy'] = s; rows.append(m)
    with open('artifacts/balance_grid.csv','w',newline='',encoding='utf-8') as f:
        cols = ['strategy','accuracy','precision','recall','f1','fp','fn','tp','tn']
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,'') for k in cols})
    print('Wrote artifacts/balance_grid.csv')
if __name__=='__main__': main()

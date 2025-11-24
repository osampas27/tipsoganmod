import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import csv, numpy as np
from tipso_gan.train import TIPSOTrainer, cfg
from tipso_gan.cicids_loader import load_cicids_csv_preset
from tipso_gan.metrics import compute_metrics, save_json, Timer
def main():
    os.makedirs('artifacts', exist_ok=True)
    Xtr,ytr,Xv,yv,Xte,yte,feats = load_cicids_csv_preset(['cicids2018.csv'])
    t = TIPSOTrainer(input_dim=Xtr.shape[1])
    with Timer() as t_train:
        t.pretrain_psogan(Xtr[ytr.flatten()==0], epochs=cfg.epochs_pretrain, batch_size=cfg.batch_size)
        t.train_tipso(Xtr[ytr.flatten()==0], Xtr, ytr, Xv, yv, epochs=cfg.epochs_tipso, batch_size=cfg.batch_size, balance_strategy='class_weight', collect_loss=True)
    with Timer() as t_test:
        preds = t.dee.predict(Xte, verbose=0)
    y_pred = preds.argmax(axis=1)
    summary, cm = compute_metrics(yte, y_pred)
    summary['train_time_s'] = round(float(t_train.elapsed), 3)
    summary['test_time_s'] = round(float(t_test.elapsed), 3)
    save_json('artifacts/perf_summary.json', summary)
    save_json('artifacts/confusion_matrix.json', {'labels':[0,1], 'matrix': cm})
    with open('artifacts/loss_history.csv','w',newline='',encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['epoch','gen_loss','disc_loss'])
        for i,(g,d) in enumerate(zip(t.history['gen_loss'], t.history['disc_loss']), start=1):
            w.writerow([i,g,d])
    print('Wrote artifacts/perf_summary.json, confusion_matrix.json, loss_history.csv')
if __name__=='__main__': main()

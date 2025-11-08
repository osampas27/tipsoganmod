import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import os, csv
from tipso_gan.train import TIPSOTrainer, cfg
from tipso_gan.cicids_loader import load_cicids_csv_preset
from tipso_gan.metrics import compute_metrics
def train_eval(use_attention: bool):
    Xtr,ytr,Xv,yv,Xte,yte,feats = load_cicids_csv_preset(['sample_cicids_small.csv'])
    Xn = Xtr[(ytr.flatten()==0)]
    t = TIPSOTrainer(input_dim=Xtr.shape[1], use_attention=use_attention)
    t.pretrain_psogan(Xn, epochs=cfg.epochs_pretrain, batch_size=cfg.batch_size)
    t.train_tipso(Xn, Xtr, ytr, Xv, yv, epochs=cfg.epochs_tipso, batch_size=cfg.batch_size, balance_strategy='class_weight')
    pred = t.dee.predict(Xte, verbose=0).argmax(axis=1)
    return compute_metrics(yte, pred)[0]
def main():
    os.makedirs('artifacts', exist_ok=True)
    rows = []
    for flag in [True, False]:
        m = train_eval(flag); m['use_attention'] = flag; rows.append(m)
    with open('artifacts/attention_ablation.csv','w',newline='',encoding='utf-8') as f:
        cols=['use_attention','accuracy','precision','recall','f1','fp','fn','tp','tn']
        w=csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,'') for k in cols})
    print('Wrote artifacts/attention_ablation.csv')
if __name__=='__main__': main()

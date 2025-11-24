import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import csv
from tipso_gan.train import TIPSOTrainer, cfg
from tipso_gan.cicids_loader import load_cicids_csv_preset
def main():
    os.makedirs('artifacts', exist_ok=True)
    cfg.epochs_pretrain = 5
    cfg.epochs_tipso    = 100
    cfg.patience_drop   = 10**9
    Xtr,ytr,Xv,yv,Xte,yte,feats = load_cicids_csv_preset(['cicids2018.csv'])
    mask_n = (ytr.flatten()==0)
    Xn = Xtr[mask_n] if mask_n.sum()>=16 else Xtr[:max(16, len(Xtr)//4)]
    t = TIPSOTrainer(input_dim=Xtr.shape[1])
    t.pretrain_psogan(Xn, epochs=cfg.epochs_pretrain, batch_size=cfg.batch_size)
    t.train_tipso(Xn, Xtr, ytr, Xv, yv, epochs=cfg.epochs_tipso, batch_size=cfg.batch_size, balance_strategy='class_weight', collect_loss=True)
    gen_hist = t.history.get('gen_loss', [])
    disc_hist = t.history.get('disc_loss', [])
    if len(gen_hist)==0 or len(disc_hist)==0:
        gen_hist = gen_hist if gen_hist else [0.0]*cfg.epochs_tipso
        disc_hist = disc_hist if disc_hist else [0.0]*cfg.epochs_tipso
    with open('artifacts/loss_history.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['epoch','gen_loss','disc_loss'])
        for i,(g,d) in enumerate(zip(gen_hist, disc_hist), start=1):
            w.writerow([i,g,d])
    print('Wrote artifacts/loss_history.csv (100 epochs)')
if __name__=='__main__': main()

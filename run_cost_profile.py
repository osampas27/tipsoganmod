import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import os, json, time
from tipso_gan.train import TIPSOTrainer, cfg
from tipso_gan.cicids_loader import load_cicids_csv_preset
def count_params(model): return int(model.count_params())
def timed_inference(model, X, repeats=3):
    model.predict(X[:128], verbose=0)
    import time
    t0=time.perf_counter()
    for _ in range(repeats):
        model.predict(X, verbose=0)
    return (time.perf_counter()-t0)/repeats
def main():
    os.makedirs('artifacts', exist_ok=True)
    Xtr,ytr,Xv,yv,Xte,yte,feats = load_cicids_csv_preset(['cicids2018.csv'])
    t = TIPSOTrainer(input_dim=Xtr.shape[1])
    t.pretrain_psogan(Xtr[ytr.flatten()==0], epochs=cfg.epochs_pretrain, batch_size=cfg.batch_size)
    t.train_tipso(Xtr[ytr.flatten()==0], Xtr, ytr, Xv, yv, epochs=cfg.epochs_tipso, batch_size=cfg.batch_size, balance_strategy='class_weight')
    params = count_params(t.dee)
    batch = Xte[:512]
    avg_s = timed_inference(t.dee, batch, repeats=3)
    with open('artifacts/cost_metrics.json','w',encoding='utf-8') as f:
        json.dump({'dee_params': params, 'avg_infer_time_per_batch_s': avg_s, 'batch_size': len(batch)}, f, indent=2)
    print('Wrote artifacts/cost_metrics.json')
if __name__=='__main__': main()

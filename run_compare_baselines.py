import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import os, json, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models
from tipso_gan.cicids_loader import load_cicids_csv_preset
from tipso_gan.metrics import compute_metrics, save_json
from tipso_gan.train import TIPSOTrainer, cfg
def build_baseline_mlp(input_dim):
    i = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu')(i)
    x = layers.Dense(128, activation='relu')(x)
    o = layers.Dense(2, activation='softmax')(x)
    return models.Model(i, o, name='BaselineMLP')
def build_tiny_transformer(input_dim):
    i = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(i)
    q = layers.Dense(64, activation='relu')(x)
    k = layers.Dense(64, activation='relu')(x)
    v = layers.Dense(64, activation='relu')(x)
    x = layers.Concatenate()([x, v])
    x = layers.Dense(128, activation='relu')(x)
    o = layers.Dense(2, activation='softmax')(x)
    return models.Model(i, o, name='TinyTransformer')
def train_and_eval(model, Xtr,ytr,Xv,yv,Xte,yte):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.fit(Xtr,ytr, validation_data=(Xv,yv), epochs=10, batch_size=cfg.batch_size, verbose=0)
    ypred = model.predict(Xte, verbose=0).argmax(axis=1)
    return compute_metrics(yte, ypred)[0]
def main():
    os.makedirs('artifacts', exist_ok=True)
    Xtr,ytr,Xv,yv,Xte,yte,feats = load_cicids_csv_preset(['sample_cicids_small.csv'])
    t = TIPSOTrainer(input_dim=Xtr.shape[1])
    t.pretrain_psogan(Xtr[ytr.flatten()==0], epochs=cfg.epochs_pretrain, batch_size=cfg.batch_size)
    t.train_tipso(Xtr[ytr.flatten()==0], Xtr, ytr, Xv, yv, epochs=cfg.epochs_tipso, batch_size=cfg.batch_size, balance_strategy='class_weight')
    tipso = compute_metrics(yte, t.dee.predict(Xte, verbose=0).argmax(axis=1))[0]
    mlp = train_and_eval(build_baseline_mlp(Xtr.shape[1]), Xtr,ytr,Xv,yv,Xte,yte)
    tiny= train_and_eval(build_tiny_transformer(Xtr.shape[1]), Xtr,ytr,Xv,yv,Xte,yte)
    save_json('artifacts/baselines_perf.json', {'TIPSO_GAN': tipso, 'BaselineMLP': mlp, 'TinyTransformer': tiny})
    print('Wrote artifacts/baselines_perf.json')
if __name__=='__main__': main()

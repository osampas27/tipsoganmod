import pandas as pd, numpy as np
def load_cicids_csv_preset(paths):
    dfs=[pd.read_csv(p) for p in paths]
    df=pd.concat(dfs,ignore_index=True)
    if 'Label' not in df.columns:
        raise ValueError('Label column missing')
    y = np.array([0 if str(s).upper()=='BENIGN' else 1 for s in df['Label'].astype(str).values], dtype=np.int32).reshape(-1,1)
    features=[c for c in df.columns if c not in ['Label','Timestamp'] and pd.api.types.is_numeric_dtype(df[c])]
    X = df[features].replace([np.inf,-np.inf],np.nan).fillna(df[features].median()).values.astype('float32')
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True)+1e-8)
    from sklearn.model_selection import train_test_split
    Xtr, Xtmp, ytr, ytmp = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    Xv, Xte, yv, yte = train_test_split(Xtmp, ytmp, train_size=0.5, random_state=42, stratify=ytmp)
    return Xtr,ytr,Xv,yv,Xte,yte,features
def load_with_attack_names(paths, label_col='Label'):
    dfs=[pd.read_csv(p) for p in paths]
    df=pd.concat(dfs,ignore_index=True)
    names = df[label_col].astype(str).values
    y = np.array([0 if str(s).upper()=='BENIGN' else 1 for s in names], dtype=np.int32)
    features=[c for c in df.columns if c not in [label_col,'Timestamp'] and pd.api.types.is_numeric_dtype(df[c])]
    X = df[features].replace([np.inf,-np.inf],np.nan).fillna(df[features].median()).values.astype('float32')
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True)+1e-8)
    return X, y.reshape(-1,1), names, features

"""
Window every .mat signal, extract engineered features (+ keep raw windows for CNN),
and save:  features.csv  (28 features + label)   and   raw_windows.npz  (X, y)
Also validate time-domain features against the provided CSV.
"""
import numpy as np, pandas as pd, glob, os, re
from scipy.io import loadmat
from features import time_features, spectral_features, envelope_features, column_names, WIN, FS

# map filename -> clean label matching the CSV's fault names
def label_from(base):
    b = base.replace(".mat","")
    if b.startswith("Time_Normal_1"): return "Normal_1"
    m = re.match(r'(B|IR|OR)(\d{3})(?:_6)?_1', b)
    typ = {"B":"Ball","IR":"IR","OR":"OR"}[m.group(1)]
    sev = m.group(2)
    return f"{typ}_{sev}_6_1" if typ=="OR" else f"{typ}_{sev}_1"

files = sorted(glob.glob('/mnt/user-data/uploads/*.mat'))
rows, raw_X, raw_y = [], [], []
tcols, scols, ecols = column_names()
allcols = tcols + scols + ecols

for f in files:
    base = os.path.basename(f); m = loadmat(f)
    num = re.search(r'_(\d+)\.mat$', base).group(1)
    sig = m[f'X{num}_DE_time'].ravel().astype(np.float64)
    rpm = float(m[f'X{num}RPM'].ravel()[0]) if f'X{num}RPM' in m else 1772.0
    fr  = rpm/60.0
    label = label_from(base)
    nwin = len(sig)//WIN
    for w in range(nwin):
        x = sig[w*WIN:(w+1)*WIN]
        feats = time_features(x) + spectral_features(x) + envelope_features(x, fr)
        rows.append(feats + [label])
        raw_X.append(x.astype(np.float32)); raw_y.append(label)
    print(f"{label:12s} {nwin:4d} windows")

df = pd.DataFrame(rows, columns=allcols + ["fault"])
df.to_csv("engineered_features.csv", index=False)
np.savez_compressed("raw_windows.npz",
                    X=np.stack(raw_X), y=np.array(raw_y))
print("\nTotal:", df.shape, "| raw:", np.stack(raw_X).shape)

# ---- VALIDATION: compare our time-domain stats to the provided CSV ----
ref = pd.read_csv('/mnt/user-data/uploads/feature_time_48k_2048_load_1.csv')
print("\n--- Validation vs provided CSV (per-class means) ---")
for col in ["rms","kurtosis","crest"]:
    a = df.groupby("fault")[col].mean()
    b = ref.groupby("fault")[col].mean()
    common = a.index.intersection(b.index)
    diff = (a[common]-b[common]).abs().mean()
    print(f"{col:9s} mean|our-csv| = {diff:.4f}")

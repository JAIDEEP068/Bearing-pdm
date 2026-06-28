"""
Classical models on engineered features.
Trains SVM / RandomForest / XGBoost and compares TWO feature sets:
  - time-only   (the 9 stats you already had in the CSV)
  - full        (time + FFT spectral + envelope-spectrum defect features)
This isolates the value added by the frequency-domain engineering.
"""
import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from features import column_names

df = pd.read_csv("engineered_features.csv")
tcols, scols, ecols = column_names()
TIME = tcols
FULL = tcols + scols + ecols

le = LabelEncoder(); y = le.fit_transform(df["fault"])
classes = le.classes_

def evaluate(featset, name):
    X = df[featset].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                           stratify=y, random_state=42)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    models = {
        "SVM (RBF)":     make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
        "RandomForest":  RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "XGBoost":       XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                                       subsample=0.9, colsample_bytree=0.9,
                                       eval_metric="mlogloss", random_state=42, n_jobs=-1),
    }
    out = {}
    for mname, mdl in models.items():
        cvsc = cross_val_score(mdl, Xtr, ytr, cv=cv, scoring="accuracy", n_jobs=-1)
        mdl.fit(Xtr, ytr)
        test_acc = accuracy_score(yte, mdl.predict(Xte))
        out[mname] = {"cv_mean": cvsc.mean(), "cv_std": cvsc.std(), "test_acc": test_acc}
        print(f"[{name:5s}] {mname:13s} CV={cvsc.mean():.4f}+/-{cvsc.std():.4f}  test={test_acc:.4f}")
    return out, (Xtr,Xte,ytr,yte)

print("="*70)
res_time, _ = evaluate(TIME, "TIME")
print("-"*70)
res_full, (Xtr,Xte,ytr,yte) = evaluate(FULL, "FULL")
print("="*70)

# Best model (XGBoost on full) -> detailed report + confusion matrix + importances
best = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                     subsample=0.9, colsample_bytree=0.9,
                     eval_metric="mlogloss", random_state=42, n_jobs=-1)
best.fit(Xtr, ytr)
yp = best.predict(Xte)
print("\nBest model = XGBoost (full features). Test report:\n")
print(classification_report(yte, yp, target_names=classes, digits=3))

# Confusion matrix
cm = confusion_matrix(yte, yp)
fig, ax = plt.subplots(figsize=(8.5,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes,
            yticklabels=classes, ax=ax, cbar=False)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("XGBoost (full features) - confusion matrix")
plt.xticks(rotation=45, ha="right"); plt.tight_layout()
plt.savefig("outputs/04_confusion_xgb.png", dpi=120); plt.close()

# Feature importance (top 15)
imp = pd.Series(best.feature_importances_, index=FULL).sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(8,6))
imp[::-1].plot(kind="barh", ax=ax, color="#55A868")
ax.set_title("XGBoost top-15 feature importances")
plt.tight_layout(); plt.savefig("outputs/05_importance.png", dpi=120); plt.close()

# Time-only vs Full comparison bar chart
fig, ax = plt.subplots(figsize=(8,4.5))
mnames = list(res_full.keys())
xpos = np.arange(len(mnames)); w=0.35
ax.bar(xpos-w/2, [res_time[m]["test_acc"] for m in mnames], w, label="time-only (9 feats)", color="#C44E52")
ax.bar(xpos+w/2, [res_full[m]["test_acc"] for m in mnames], w, label="full (28 feats)", color="#4C72B0")
ax.set_xticks(xpos); ax.set_xticklabels(mnames); ax.set_ylim(0.8,1.0)
ax.set_ylabel("test accuracy"); ax.legend(); ax.set_title("Value of FFT + envelope features")
for i,m in enumerate(mnames):
    ax.text(i-w/2, res_time[m]["test_acc"]+.003, f"{res_time[m]['test_acc']:.3f}", ha="center", fontsize=8)
    ax.text(i+w/2, res_full[m]["test_acc"]+.003, f"{res_full[m]['test_acc']:.3f}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig("outputs/06_feature_value.png", dpi=120); plt.close()

# save metrics + best model
json.dump({"time_only":res_time, "full":res_full}, open("outputs/metrics_classical.json","w"), indent=2)
import joblib; joblib.dump({"model":best,"label_encoder":le,"features":FULL}, "outputs/xgb_model.joblib")
print("\nSaved plots, metrics, model.")

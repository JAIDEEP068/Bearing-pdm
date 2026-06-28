"""
1D-CNN on RAW vibration windows (2048 samples).
The convolutional filters learn local waveform/impulse patterns directly from
the time series -- no manual FFT/feature engineering. This is the honest
deep-learning counterpart to the classical pipeline.
"""
import numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
tf.random.set_seed(42); np.random.seed(42)

d = np.load("raw_windows.npz", allow_pickle=True)
X, y_raw = d["X"], d["y"]
le = LabelEncoder(); y = le.fit_transform(y_raw); classes = le.classes_

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# global standardization (fit on TRAIN only) -- preserves relative amplitude
mu, sd = Xtr.mean(), Xtr.std()
Xtr = ((Xtr-mu)/sd)[...,None]; Xte = ((Xte-mu)/sd)[...,None]

def build(n_cls):
    m = models.Sequential([
        layers.Input((2048,1)),
        layers.Conv1D(16, 64, strides=2, activation="relu", padding="same"),
        layers.BatchNormalization(), layers.MaxPool1D(2),
        layers.Conv1D(32, 16, activation="relu", padding="same"),
        layers.BatchNormalization(), layers.MaxPool1D(2),
        layers.Conv1D(64, 8, activation="relu", padding="same"),
        layers.BatchNormalization(), layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"), layers.Dropout(0.3),
        layers.Dense(n_cls, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

model = build(len(classes))
print(f"Params: {model.count_params():,}")
es = callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy")
hist = model.fit(Xtr, ytr, validation_split=0.15, epochs=60, batch_size=32,
                 callbacks=[es], verbose=0)
test_acc = accuracy_score(yte, model.predict(Xte, verbose=0).argmax(1))
print(f"Epochs run: {len(hist.history['loss'])}  |  CNN test accuracy: {test_acc:.4f}")
yp = model.predict(Xte, verbose=0).argmax(1)
print("\n", classification_report(yte, yp, target_names=classes, digits=3))

# training curve
fig, ax = plt.subplots(1,2, figsize=(11,4))
ax[0].plot(hist.history["accuracy"], label="train"); ax[0].plot(hist.history["val_accuracy"], label="val")
ax[0].set_title("CNN accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(hist.history["loss"], label="train"); ax[1].plot(hist.history["val_loss"], label="val")
ax[1].set_title("CNN loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout(); plt.savefig("outputs/07_cnn_training.png", dpi=120); plt.close()

cm = confusion_matrix(yte, yp)
fig, ax = plt.subplots(figsize=(8.5,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=classes, yticklabels=classes, ax=ax, cbar=False)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"1D-CNN confusion matrix (test acc {test_acc:.3f})")
plt.xticks(rotation=45, ha="right"); plt.tight_layout()
plt.savefig("outputs/08_confusion_cnn.png", dpi=120); plt.close()

json.dump({"cnn_test_acc": float(test_acc), "epochs": len(hist.history["loss"]),
           "params": int(model.count_params())}, open("outputs/metrics_cnn.json","w"), indent=2)
model.save("outputs/cnn_model.keras")
print("Saved CNN model, plots, metrics.")

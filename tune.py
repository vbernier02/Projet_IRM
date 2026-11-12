"""Étape 5 : optimisation bayésienne (learning rate, batch size, filtres, dropout) + comparaison.
Usage : python tune.py [data_dir]"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras_tuner as kt
from tensorflow import keras
from common import load_data, batched, build_model
from train import run

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
train, val, _ = load_data(DATA)


class HM(kt.HyperModel):
    def build(self, hp):
        return build_model(filters=hp.Choice("filters", [16, 32, 48]),
                           dropout=hp.Float("dropout", 0.2, 0.6, step=0.1),
                           lr=hp.Float("lr", 1e-4, 1e-2, sampling="log"))

    def fit(self, hp, model, *args, **kwargs):
        bs = hp.Choice("batch_size", [16, 32, 64])
        return model.fit(batched(train, bs, True), validation_data=batched(val, bs), **kwargs)


tuner = kt.BayesianOptimization(HM(), objective="val_accuracy", max_trials=8,
                                directory="tuning", project_name="brain_mri", overwrite=True)
tuner.search(epochs=12, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=4)])
best = tuner.get_best_hyperparameters(1)[0].values
print("Meilleurs hyperparamètres :", best)

# Ré-entraînement complet avec les meilleurs hyperparamètres
run("tuned", filters=best["filters"], dropout=best["dropout"], lr=best["lr"],
    batch_size=best["batch_size"], data_dir=DATA)

# Comparaison baseline vs optimisé
M = {n: json.load(open(f"models/{n}/metrics.json")) for n in ["baseline", "tuned"]}
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for n, m in M.items():
    ax[0].plot(m["history"]["val_accuracy"], label=n); ax[1].plot(m["history"]["val_loss"], label=n)
ax[0].set_title("val_accuracy"); ax[1].set_title("val_loss")
for a in ax: a.legend(); a.grid(alpha=.3); a.set_xlabel("epoch")
plt.tight_layout(); plt.savefig("comparison.png", dpi=120)
print(f"{'modèle':10s} {'test acc':>9s} {'test loss':>10s} {'macro-F1':>9s} {'std val_acc (5 dern.)':>22s}")
for n, m in M.items():
    print(f"{n:10s} {m['test_accuracy']:9.4f} {m['test_loss']:10.4f} "
          f"{m['report']['macro avg']['f1-score']:9.4f} {np.std(m['history']['val_accuracy'][-5:]):22.4f}")

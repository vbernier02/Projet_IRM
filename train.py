"""Étapes 3-4 : entraînement + évaluation. Usage : python train.py [data_dir]"""
import sys, json, io
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from common import CLASSES, IMG_SIZE, load_data, batched, build_model


def plot_curves(h, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, m in zip(ax, ["accuracy", "loss"]):
        a.plot(h[m], label="train"); a.plot(h["val_" + m], label="validation")
        a.set_title(m); a.set_xlabel("epoch"); a.legend(); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def run(name="baseline", filters=32, dropout=0.4, lr=1e-3, batch_size=32, epochs=40, data_dir="data"):
    out = Path("models") / name
    out.mkdir(parents=True, exist_ok=True)
    train, val, test = load_data(data_dir)
    train_b, val_b, test_b = batched(train, batch_size, True), batched(val, batch_size), batched(test, batch_size)

    model = build_model(filters, dropout, lr)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(out / "model.keras", monitor="val_loss", save_best_only=True),
    ]
    hist = model.fit(train_b, validation_data=val_b, epochs=epochs, callbacks=callbacks)
    h = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    model.save(out / "model.keras")  # poids restaurés = meilleure époque
    plot_curves(h, out / "curves.png")

    # Évaluation sur le test
    test_loss, test_acc = model.evaluate(test_b, verbose=0)
    y_true = np.concatenate([y for _, y in test_b])
    y_pred = np.argmax(model.predict(test_b, verbose=0), axis=1)
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap="Blues", xticks_rotation=30)
    plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=120); plt.close()

    buf = io.StringIO(); model.summary(print_fn=lambda s: buf.write(s + "\n"))
    metrics = {
        "test_loss": float(test_loss), "test_accuracy": float(test_acc),
        "report": report, "confusion_matrix": cm.tolist(),
        "params": {"filters_base": filters, "dropout": dropout, "learning_rate": lr,
                   "batch_size": batch_size, "epochs_run": len(h["loss"]), "img_size": IMG_SIZE,
                   "n_parameters": int(model.count_params()), "optimizer": "Adam",
                   "kernel_size": "3x3",    "architecture": "5 x [Conv3x3-ReLU-MaxPool] + Flatten + Dense(128) + Dropout + Softmax"},
        "summary": buf.getvalue(), "history": h,
    }
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    print(f"[{name}] test acc = {test_acc:.4f} | test loss = {test_loss:.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    return metrics


if __name__ == "__main__":
    run(data_dir=sys.argv[1] if len(sys.argv) > 1 else "data")

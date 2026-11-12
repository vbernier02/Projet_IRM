"""Grad-CAM : heatmap des zones qui ont influencé la prédiction."""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import colormaps
from PIL import Image
from common import LAST_CONV_ACT, IMG_SIZE


def preprocess(pil_img):
    """Même prétraitement que l'entraînement : niveaux de gris, 128x128 (la normalisation est dans le modèle)."""
    img = pil_img.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype="float32")[None, ..., None]

def make_gradcam(model, x, class_idx=None, layer_name=LAST_CONV_ACT):
    x = tf.convert_to_tensor(x)
    with tf.GradientTape() as tape:
        h, conv = x, None
        #Correction bug : make_gradcam ne parcours pas les couches de Sequential
        for layer in model.layers:#forward pass couche par couche
            h = layer(h, training=False)
            if layer.name == layer_name:
                conv = h#sortie de la dernière conv (après ReLU)
        preds = h
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2))                     # importance de chaque filtre
    cam = tf.nn.relu(tf.reduce_sum(conv * weights[:, None, None, :], axis=-1))[0]
    cam = (cam / (tf.reduce_max(cam) + 1e-8)).numpy()
    return cam, preds[0].numpy()


def overlay(pil_img, cam, alpha=0.4):
    base = pil_img.convert("L").resize((IMG_SIZE * 3, IMG_SIZE * 3))
    heat = Image.fromarray(np.uint8(cam * 255)).resize(base.size, Image.BICUBIC)
    heat = (colormaps["jet"](np.asarray(heat) / 255.0)[..., :3] * 255).astype("uint8")
    base_rgb = np.asarray(base.convert("RGB"))
    return Image.fromarray(((1 - alpha) * base_rgb + alpha * heat).astype("uint8"))

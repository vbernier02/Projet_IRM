"""Constantes, chargement des données et définition du CNN."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Rescaling, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 128
AUTOTUNE = tf.data.AUTOTUNE


def load_data(data_dir="data", val_split=0.15, seed=9):
    """Training/ -> train + validation ; Testing/ -> test (jamais utilisé pour l'entraînement)."""
    kw = dict(image_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale",
              label_mode="int", class_names=CLASSES, batch_size=None, seed=seed)
    train = keras.utils.image_dataset_from_directory(
        f"{data_dir}/Training", validation_split=val_split, subset="training", shuffle=True, **kw)
    val = keras.utils.image_dataset_from_directory(
        f"{data_dir}/Training", validation_split=val_split, subset="validation", shuffle=True, **kw)
    test = keras.utils.image_dataset_from_directory(f"{data_dir}/Testing", shuffle=False, **kw)
    return train.cache(), val.cache(), test.cache()


def batched(ds, batch_size, shuffle=False):
    if shuffle:
        ds = ds.shuffle(2000)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def build_model(filters=32, dropout=0.4, lr=1e-3, dense_units=128):
    #CNN : 5 blocs [Conv3x3 (ReLU) -> MaxPool], Flatten, Dense, Dropout, Softmax.
    #Filtres 32 → 64 → 128 → 256 : on double à chaque bloc pour compenser la réduction de taille par le pooling
    #Noyaux 3×3 : Taille standards, et plusieurs petits noyaux empilés couvrent un grand champ visuel avec peu de paramètres
    #Dropout 0.5 : il désactive aléatoirement la moitié des neurones de la couche Dense pour limite un potentiel surapprentissage

    cnn = Sequential()
    cnn.add(Input(shape=(IMG_SIZE, IMG_SIZE, 1)))
    cnn.add(Rescaling(1. / 255))  # normalisation [0,1] intégrée au modèle

    cnn.add(Conv2D(filters, (3, 3), activation="relu"))
    cnn.add(MaxPooling2D((2, 2)))

    cnn.add(Conv2D(filters * 2, (3, 3), activation="relu"))
    cnn.add(MaxPooling2D((2, 2)))

    cnn.add(Conv2D(filters * 4, (3, 3), activation="relu"))
    cnn.add(MaxPooling2D((2, 2)))

    cnn.add(Conv2D(filters * 8, (3, 3), activation="relu"))
    cnn.add(MaxPooling2D((2, 2)))

    cnn.add(Conv2D(filters * 8, (3, 3), activation="relu", name="last_conv"))
    cnn.add(MaxPooling2D((2, 2)))

    cnn.add(Flatten())
    cnn.add(Dense(dense_units, activation="relu"))
    cnn.add(Dropout(dropout))
    cnn.add(Dense(len(CLASSES), activation="softmax"))

    cnn.compile(optimizer=Adam(learning_rate=lr),
                loss="sparse_categorical_crossentropy",  # étiquettes entières (label_mode="int")
                metrics=["accuracy"])
    return cnn


LAST_CONV_ACT = "last_conv"  # dernière couche convolutive (pour Grad-CAM)

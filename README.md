# Classification de tumeurs cérébrales (IRM)

1. `pip install -r requirements.txt`
2. Télécharger le dataset Kaggle (masoudnickparvar/brain-tumor-mri-dataset) et le placer dans `data/`
   (structure attendue : `data/Training/<classe>/` et `data/Testing/<classe>/`).
3. `python explore.py data`   -> statistiques + `exploration.png`
4. `python train.py data`     -> modèle de référence dans `models/baseline/`
5. `python tune.py data`      -> optimisation bayésienne + `models/tuned/` + `comparison.png`
6. `streamlit run app.py`

## Choix à justifier à l'oral
- **128x128, niveaux de gris** : IRM mono-canal ; 128 px garde les contours des lésions tout en restant rapide.
- **Normalisation /255** dans le modèle (même prétraitement à l'entraînement et en inférence).
- **Split** : 85 % / 15 % du dossier Training pour train/validation ; Testing réservé à l'évaluation finale.
- **4 blocs Conv3x3 + BN + ReLU + MaxPool**, filtres f, 2f, 4f, 8f ; GAP (moins de surapprentissage que Flatten).
- **Callbacks** : EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
- **Surapprentissage** : comparer l'écart train/val sur `curves.png`.
- **Grad-CAM** : gradient du score de classe par rapport aux cartes de la dernière conv (`relu_4`).


## Modif Vincent pour les choix
128×128 : les images d’origine ont des tailles variables, il faut une taille fixe.
Niveaux de gris : une IRM est monochrome, mais certaine avec des teintes de differente, autant tout mettre au gris pour ne pas induire de biais.
Normalisation 1/255 : ramène les pixels dans [0, 1] pour stabiliser l’entraînement.

Filtres 32 → 64 → 128 → 256 : on double à chaque bloc pour compenser la réduction de taille par le pooling
Noyaux 3×3 : Taille standards, et plusieurs petits noyaux empilés couvrent un grand champ visuel avec peu de paramètres
Dropout 0.5 : il désactive aléatoirement la moitié des neurones de la couche Dense pour limite un potentiel surapprentissage

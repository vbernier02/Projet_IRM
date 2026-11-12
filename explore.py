"""Étape 1 : exploration du dataset."""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from common import CLASSES

data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for split in ["Training", "Testing"]:
    print(split, {c: len(list((data_dir / split / c).glob("*"))) for c in CLASSES})
sizes = Counter()
for c in CLASSES:
    files = sorted((data_dir / "Training" / c).glob("*"))
    for f in files[:200]:
        sizes[Image.open(f).size] += 1
print("Tailles les plus fréquentes (échantillon) :", sizes.most_common(5))
for j, c in enumerate(CLASSES):
    files = sorted((data_dir / "Training" / c).glob("*"))
    for i in range(2):
        axes[i, j].imshow(Image.open(files[i * 50]).convert("L"), cmap="gray")
        axes[i, j].set_title(c); axes[i, j].axis("off")
plt.tight_layout(); plt.savefig("exploration.png", dpi=120)

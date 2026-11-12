"""Application Streamlit. Lancer : streamlit run app.py"""
import io, json, tempfile
from pathlib import Path
import numpy as np, pandas as pd
import streamlit as st
from PIL import Image
from fpdf import FPDF
from common import CLASSES

st.set_page_config(page_title="Tumeurs cérébrales - IRM", layout="wide")
page = st.sidebar.radio("Navigation", ["1. Exploration des données", "2. Modèle", "3. Test sur une image", "4. Rapport PDF"])
data_dir = Path(st.sidebar.text_input("Dossier du dataset", "data"))
model_dirs = sorted(p.parent.name for p in Path("models").glob("*/model.keras"))
model_name = st.sidebar.selectbox("Modèle", model_dirs) if model_dirs else None


@st.cache_resource
def get_model(name):
    from tensorflow import keras
    return keras.models.load_model(f"models/{name}/model.keras")


def get_metrics(name):
    return json.load(open(f"models/{name}/metrics.json"))


def list_images(split, cls):
    return sorted((data_dir / split / cls).glob("*"))


# ---------------------------------------------------------------- 1. Exploration
if page.startswith("1"):
    st.title("Exploration des données")
    c1, c2 = st.columns(2)
    split = c1.selectbox("Ensemble", ["Training", "Testing"])
    cls = c2.selectbox("Classe", CLASSES)
    files = list_images(split, cls)
    if not files:
        st.warning(f"Aucune image dans {data_dir / split / cls}")
    else:
        st.caption(f"{len(files)} images")
        per_page = 12
        p = st.number_input("Page", 1, max(1, (len(files) - 1) // per_page + 1), 1)
        chunk = files[(p - 1) * per_page: p * per_page]
        cols = st.columns(4)
        for i, f in enumerate(chunk):
            cols[i % 4].image(Image.open(f), caption=f.name, use_container_width=True)
        sel = st.selectbox("Agrandir l'image", [f.name for f in chunk])
        st.image(Image.open(data_dir / split / cls / sel), width=700)

# ---------------------------------------------------------------- 2. Modèle
elif page.startswith("2"):
    st.title("Modèle")
    if not model_name:
        st.error("Aucun modèle dans models/. Lancez d'abord train.py."); st.stop()
    m = get_metrics(model_name)
    st.subheader("Paramètres"); st.json(m["params"])
    with st.expander("Architecture (model.summary)"):
        st.code(m["summary"])
    st.subheader("Métriques sur l'ensemble de test")
    a, b = st.columns(2)
    a.metric("Accuracy", f"{m['test_accuracy']:.4f}"); b.metric("Loss", f"{m['test_loss']:.4f}")
    st.dataframe(pd.DataFrame(m["report"]).T.round(3))
    c1, c2 = st.columns(2)
    c1.image(f"models/{model_name}/confusion_matrix.png", caption="Matrice de confusion")
    c2.image(f"models/{model_name}/curves.png", caption="Courbes d'apprentissage")

# ---------------------------------------------------------------- 3. Test image
elif page.startswith("3"):
    st.title("Test sur une image")
    if not model_name:
        st.error("Aucun modèle disponible."); st.stop()
    from gradcam import preprocess, make_gradcam, overlay
    src = st.radio("Source", ["Image du dataset", "Téléverser"], horizontal=True)
    img, true_label = None, None
    if src == "Téléverser":
        up = st.file_uploader("Image IRM", type=["jpg", "jpeg", "png"])
        if up: img = Image.open(up)
    else:
        c1, c2 = st.columns(2)
        cls = c1.selectbox("Classe", CLASSES)
        files = list_images("Testing", cls)
        if files:
            f = c2.selectbox("Fichier", files, format_func=lambda p: p.name)
            img, true_label = Image.open(f), cls
    if img is not None:
        model = get_model(model_name)
        cam, probs = make_gradcam(model, preprocess(img))
        pred = int(np.argmax(probs))
        heat = overlay(img, cam)
        c1, c2, c3 = st.columns(3)
        c1.image(img.convert("L"), caption="Image testée", use_container_width=True)
        c2.image(heat, caption="Grad-CAM", use_container_width=True)
        c3.subheader(f"Prédiction : {CLASSES[pred]}")
        if true_label: c3.caption(f"Classe réelle : {true_label}")
        c3.bar_chart(pd.DataFrame({"probabilité": probs}, index=CLASSES))
        c3.dataframe(pd.DataFrame({"classe": CLASSES, "probabilité": (probs * 100).round(2)}))
        st.session_state["last"] = dict(img=img, heat=heat, probs=probs, pred=pred, model=model_name)

# ---------------------------------------------------------------- 4. PDF
else:
    st.title("Rapport PDF")
    last = st.session_state.get("last")
    if not last:
        st.info("Testez d'abord une image (page 3)."); st.stop()

    def L(t):  # polices de base fpdf = latin-1
        return str(t).encode("latin-1", "replace").decode("latin-1")

    def build_pdf(last):
        m = get_metrics(last["model"])
        pdf = FPDF(); pdf.add_page()
        nl = dict(new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 16); pdf.cell(0, 10, L("Rapport d'analyse IRM cérébrale"), **nl)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, L(f"Modèle : {last['model']} | Prédiction : {CLASSES[last['pred']]}"), **nl)
        y = pdf.get_y() + 2
        for k, (im, cap) in enumerate([(last["img"].convert("L").convert("RGB"), "Image testée"), (last["heat"], "Grad-CAM")]):
            buf = io.BytesIO(); im.resize((384, 384)).save(buf, "PNG"); buf.seek(0)
            pdf.image(buf, x=10 + k * 95, y=y, w=85)
            pdf.set_xy(10 + k * 95, y + 86); pdf.cell(85, 5, L(cap), align="C")
        pdf.set_y(y + 95)
        pdf.set_font("Helvetica", "B", 12); pdf.cell(0, 8, L("Probabilités de prédiction"), **nl)
        pdf.set_font("Helvetica", "", 10)
        for c, p in zip(CLASSES, last["probs"]):
            pdf.cell(0, 6, L(f"  {c} : {p*100:.2f} %"), **nl)
        pdf.ln(3); pdf.set_font("Helvetica", "B", 12); pdf.cell(0, 8, L("Métriques du modèle (ensemble de test)"), **nl)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, L(f"  Accuracy : {m['test_accuracy']:.4f} | Loss : {m['test_loss']:.4f}"), **nl)
        for c in CLASSES:
            r = m["report"][c]
            pdf.cell(0, 6, L(f"  {c}: precision {r['precision']:.3f}, rappel {r['recall']:.3f}, F1 {r['f1-score']:.3f}"), **nl)
        pdf.ln(3); pdf.set_font("Helvetica", "B", 12); pdf.cell(0, 8, L("Paramètres du modèle"), **nl)
        pdf.set_font("Helvetica", "", 10)
        for k, v in m["params"].items():
            pdf.multi_cell(0, 6, L(f"  {k} : {v}"), **nl)
        pdf.ln(4); pdf.set_font("Helvetica", "I", 8)
        pdf.multi_cell(0, 4, L("Outil d'aide à la décision à visée pédagogique : ne remplace pas l'avis d'un praticien."))
        return bytes(pdf.output())

    st.success(f"Dernière image testée : {CLASSES[last['pred']]}")
    st.download_button("Télécharger le rapport PDF", build_pdf(last), "rapport_irm.pdf", "application/pdf")

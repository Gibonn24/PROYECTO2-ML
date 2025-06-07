import streamlit as st, pandas as pd, numpy as np, requests, joblib
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
import tempfile, os
from utils.funciones import extract_features     # tu extractor CNN-HOG...

TMDB_KEY = st.secrets["api"]["tmdb_key"]

# ---------- CARGA LIGERA DESDE DRIVE ----------
@st.cache_data
def load_dataset():
    # Ruta al pkl local:  PROYECTO2-ML/data/dataset_features_pca.pkl
    pkl_path = os.path.join(os.path.dirname(__file__), "data", "dataset_features_pca.pkl")
    return joblib.load(pkl_path)

ds            = load_dataset()
X_pca         = ds["X_pca"]          # (n, 50)
imdb_ids      = ds["imdbId"]         # (n,)
labels        = ds["labels"]         # (n,)
centers       = ds["centers"]        # (k, 50)
pca_mean      = ds["pca_mean"]
pca_comp      = ds["pca_components"]

# ----- helpers -----
def project_to_pca(feat_5700):
    return (feat_5700 - pca_mean) @ pca_comp.T        # (1,50)

def get_title(imdb_id):
    imdb_tt = f"tt{int(imdb_id):07d}"
    url = f"https://api.themoviedb.org/3/find/{imdb_tt}?api_key={TMDB_KEY}&external_source=imdb_id"
    r = requests.get(url).json()
    for key in ("movie_results", "tv_results"):
        if r.get(key): return r[key][0].get("title") or r[key][0].get("name")
    return "Título no disponible"

def show_poster(imdb_id, cap=None):
    imdb_tt = f"tt{int(imdb_id):07d}"
    url = f"https://api.themoviedb.org/3/find/{imdb_tt}?api_key={TMDB_KEY}&external_source=imdb_id"
    r = requests.get(url).json()
    for key in ("movie_results", "tv_results"):
        if r.get(key) and r[key][0].get("poster_path"):
            img = "https://image.tmdb.org/t/p/w500" + r[key][0]["poster_path"]
            st.image(img, caption=cap, use_container_width=True); return
    st.write("🖼️ sin póster")

# ---------- UI ----------
st.title("🎬 Recomendador visual de películas")

# 1) Consulta por IMDb ID
imdb_input = st.text_input("🔎 IMDb ID (ej. 114709):")
if imdb_input:
    try:
        qid = int(imdb_input)
        if qid not in imdb_ids:
            st.error("ID no encontrado"); st.stop()
        idx   = np.where(imdb_ids == qid)[0][0]
        vec50 = X_pca[idx].reshape(1, -1)
        cl    = labels[idx]
        mask  = labels == cl
        sims  = cosine_similarity(vec50, X_pca[mask])[0]
        top   = sims.argsort()[-6:][::-1]   # base + 5
        top_ids = imdb_ids[mask][top]
        st.subheader(f"Película base ({get_title(qid)}) y similares:")
        cols = st.columns(len(top_ids))
        for c, mid in zip(cols, top_ids):
            with c: show_poster(mid, get_title(mid))
    except ValueError:
        st.warning("Ingrese un número entero válido.")

st.divider()

# 2) Consulta por imagen
uploaded = st.file_uploader("📤 Sube un póster (.jpg/.png) para recomendaciones", type=["jpg","jpeg","png"])
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded.read()); path = tmp.name
    feat5700 = extract_features(path).reshape(1, -1)
    vec50    = project_to_pca(feat5700)
    cl       = np.argmin(np.linalg.norm(centers - vec50, axis=1))
    mask     = labels == cl
    sims     = cosine_similarity(vec50, X_pca[mask])[0]
    top      = sims.argsort()[-5:][::-1]
    rec_ids  = imdb_ids[mask][top]
    st.success(f"La imagen se clasificó en clúster {cl}. Películas sugeridas:")
    cols = st.columns(5)
    for c, mid in zip(cols, rec_ids):
        with c: show_poster(mid, get_title(mid))
    os.remove(path)

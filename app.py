import traceback
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.decomposition import PCA, TruncatedSVD
from collections import deque
import seaborn as sns
import warnings
import os
import tempfile
import requests
import glob
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from utils.funciones import extract_features, kmeans, dbscan
import sys
# Patch for legacy pickled objects referencing deprecated numpy paths
sys.modules.setdefault('numpy._core', np.core)
sys.modules.setdefault('numpy._core.numeric', np.core.numeric)

try:
    st.write("✅ La app arrancó...")
    # tus imports y funciones aqu
    st.title("Análisis de Componentes Principales (PCA & SVD)")

    # Ruta local al archivo .pkl

    @st.cache_data
    def load_data():
        # Leer y juntar todos los .pkl divididos
        chunk_paths = sorted(glob.glob("data/features_part_*.pkl"))
        dataframes = [pd.read_pickle(path) for path in chunk_paths]
        df_total = pd.concat(dataframes, ignore_index=True)
        return df_total

    features_df = load_data()

    TMDB_API_KEY = st.secrets["api"]["tmdb_key"]

    filtered_df = pd.read_csv("data/filtered_df.csv")

    # Mostrar columnas y shape
    st.write("📐 Dimensiones:", features_df.shape)

    # Separar datos numéricos
    X = features_df.drop(columns=['imdbId'])

    # Aplicar PCA
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X)

    # Aplicar SVD
    svd = TruncatedSVD(n_components=50)
    X_svd = svd.fit_transform(X)

    # Visualización: Varianza acumulada explicada por PCA
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', label='PCA')
    ax.set_title('PCA: Varianza Acumulada Explicada')
    ax.set_xlabel('Número de Componentes')
    ax.set_ylabel('Varianza Acumulada')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Mostrar la varianza total explicada
    varianza_total = np.sum(pca.explained_variance_ratio_)
    st.write(f"🔍 **Varianza total explicada por PCA (50 componentes):** {varianza_total:.4f}")

    # Visualización: Varianza acumulada aproximada por SVD
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(np.cumsum(svd.explained_variance_ratio_), marker='x', color='orange', label='SVD')
    ax2.set_title('SVD: Varianza Acumulada Aproximada')
    ax2.set_xlabel('Número de Componentes')
    ax2.set_ylabel('Varianza Acumulada')
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

    # Mostrar la varianza total explicada por SVD
    varianza_svd = np.sum(svd.explained_variance_ratio_)
    st.write(f"🔍 **Varianza acumulada por SVD (50 componentes):** {varianza_svd:.4f}")


    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    st.header("Evaluación de Clustering y Visualización con UMAP")

    # Dataset para clustering (usa X_pca, o X_svd si prefieres)
    X_clust = X_pca

    # -------- K-means desde cero --------
    k = st.slider("Número de clusters para K-means++", 2, 20, 10)
    labels_kmeans, centers = kmeans(X_clust, k, random_state=42)

    silhouette_kmeans = silhouette_score(X_clust, labels_kmeans)
    st.subheader("🔵 K-means++")
    st.write(f"**Silhouette Score:** {silhouette_kmeans:.4f}")

    # Verificar si tienes la columna 'genres' para calcular NMI
    if 'genres' in filtered_df.columns:
        nmi_val = normalized_mutual_info_score(filtered_df['genres'], labels_kmeans, average_method='arithmetic')
        st.write(f"**NMI con 'genres':** {nmi_val:.4f}")
    else:
        st.info("📌 La columna 'genres' no está disponible para calcular NMI.")

    # -------- DBSCAN desde cero --------
    eps = st.slider("Epsilon (DBSCAN)", 0.1, 10.0, 2.5, 0.1)
    min_samples = st.slider("Min Samples (DBSCAN)", 1, 20, 5)

    labels_dbscan = dbscan(X_clust, eps=eps, min_samples=min_samples)

    silhouette_dbscan = silhouette_score(X_clust, labels_dbscan)
    n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    n_noise = list(labels_dbscan).count(-1)

    st.subheader("🟠 DBSCAN")
    st.write(f"**Silhouette Score:** {silhouette_dbscan:.4f}")
    st.write(f"**Clústeres encontrados (sin contar ruido):** {n_clusters_dbscan}")
    st.write(f"**Puntos marcados como ruido:** {n_noise}")

    # -------- UMAP para visualización --------
    st.subheader("🌐 Visualización UMAP")

    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_clust)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=labels_kmeans, palette='tab10', ax=axs[0], s=10)
    axs[0].set_title("K-means++ (UMAP)")

    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=labels_dbscan, palette='Set1', ax=axs[1], s=10)
    axs[1].set_title("DBSCAN (UMAP)")

    plt.tight_layout()
    st.pyplot(fig)

    from sklearn.manifold import TSNE

    st.header("📉 Visualización 2D de Clústeres (PCA o t-SNE)")

    # Elegir qué etiquetas usar
    label_opcion = st.selectbox("¿Qué etiquetas de clustering quieres visualizar?", ['K-means++', 'DBSCAN'])

    # Definir X y labels
    X = X_pca  # o usa X_svd si prefieres
    labels = labels_kmeans if label_opcion == 'K-means++' else labels_dbscan

    # Elegir método de proyección
    proyeccion = st.radio("Método de reducción de dimensión", ["PCA (rápido)", "t-SNE (más expresivo, lento)"])

    if proyeccion == "PCA (rápido)":
        pca_2d = PCA(n_components=2)
        X_2d = pca_2d.fit_transform(X)
    else:
        st.info("⏳ Ejecutando t-SNE... esto puede tardar unos segundos")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_2d = tsne.fit_transform(X)

    # Visualización
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=labels, palette='tab10', s=15, linewidth=0, ax=ax)
    ax.set_title(f"Visualización 2D de los clústeres ({label_opcion} + {proyeccion})")
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.legend(title="Clúster", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    st.header("🎬 Recomendador de películas por similitud visual (por clúster)")

    # Asegurarse de que el DataFrame tiene 'cluster'
    features_df['cluster'] = labels_kmeans


    # Mostrar póster en Streamlit

    def show_poster(imdb_id, caption=None):
        imdb_id_formatted = f'tt{int(imdb_id):07d}'
        url = f'https://api.themoviedb.org/3/find/{imdb_id_formatted}?api_key={TMDB_API_KEY}&external_source=imdb_id'

        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for result_set in ['movie_results', 'tv_results']:
                results = data.get(result_set, [])
                if results:
                    poster_path = results[0].get('poster_path')
                    if poster_path:
                        poster_url = f'https://image.tmdb.org/t/p/w500{poster_path}'
                        st.image(poster_url, caption=caption, use_container_width=True)
                        return
        st.warning(f"🖼️ No se encontró el póster para IMDb ID: {imdb_id}")
    # Recomendador

    def get_title_from_imdb(imdb_id):
        imdb_id_formatted = f'tt{int(imdb_id):07d}'
        url = f'https://api.themoviedb.org/3/find/{imdb_id_formatted}?api_key={TMDB_API_KEY}&external_source=imdb_id'
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            movie = data.get('movie_results', [])
            tv = data.get('tv_results', [])
            if movie:
                return movie[0].get('title', 'Título no disponible')
            elif tv:
                return tv[0].get('name', 'Título no disponible')
        return 'Título no disponible'
    features_df.set_index('imdbId', inplace=True)
    def recomendar_similares(imdb_id_consulta, top_n=5):
        if imdb_id_consulta not in features_df.index:
            st.error("❌ imdbId no encontrado.")
            return

        query_vector = features_df.loc[imdb_id_consulta].drop(['cluster']).values.reshape(1, -1)
        clust_id = features_df.loc[imdb_id_consulta]['cluster']

        same_cluster = features_df[features_df['cluster'] == clust_id]
        similarities = cosine_similarity(query_vector, same_cluster.drop(columns='cluster').values)[0]

        top_indices = similarities.argsort()[::-1][1:top_n+1]
        top_ids = same_cluster.iloc[top_indices].index

        titulo = get_title_from_imdb(imdb_id_consulta)
        st.subheader(f"🎥 Película base: {titulo} (Clúster {clust_id})")
        show_poster(imdb_id_consulta, caption="Película Base")


        st.markdown("### 🔍 Películas más similares visualmente:")
        cols = st.columns(top_n)

        for i, imdb_id in enumerate(top_ids):
            with cols[i]:
                titulo = get_title_from_imdb(imdb_id)
                show_poster(imdb_id, caption=titulo)


    # Interfaz de usuario
    user_input = st.text_input("🔎 Ingresa un imdbId válido para recomendar similares:")
    if user_input:
        try:
            imdb_id_input = int(user_input)
            recomendar_similares(imdb_id_input, top_n=5)
        except ValueError:
            st.warning("❗ Ingrese un imdbId numérico válido (ej. 114709).")


    st.header("🖼️ Recomendación basada en una imagen subida")

    uploaded_file = st.file_uploader("📤 Sube una imagen (póster) para encontrar películas similares", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Mostrar imagen subida
        image_bytes = uploaded_file.read()
        st.image(image_bytes, caption="Imagen subida", use_container_width=True)

        # Guardar en un archivo temporal para poder usar extract_features
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = tmp_file.name

        with st.spinner("🔍 Extrayendo características visuales..."):
            try:
                # Extraer características con tu función
                features_img = extract_features(tmp_path).reshape(1, -1)

                # Aplicar misma reducción de dimensionalidad que el dataset original
                features_img_pca = pca.transform(features_img)

                # Asignar al clúster más cercano (K-means++)
                dists = np.linalg.norm(centers - features_img_pca, axis=1)
                assigned_cluster = np.argmin(dists)

                st.success(f"La imagen fue asignada al clúster {assigned_cluster} 🎯")

                # Buscar películas similares dentro del mismo clúster
                same_cluster = features_df[features_df['cluster'] == assigned_cluster]
                similarities = cosine_similarity(features_img, same_cluster.drop(columns='cluster').values)[0]
                top_indices = similarities.argsort()[::-1][:5]
                top_ids = same_cluster.iloc[top_indices].index

                st.markdown("### 🎬 Películas recomendadas para ti:")
                cols = st.columns(5)
                for i, imdb_id in enumerate(top_ids):
                    with cols[i]:
                        titulo = get_title_from_imdb(imdb_id)
                        show_poster(imdb_id, caption=titulo)

            except Exception as e:
                st.error(f"❌ Ocurrió un error al procesar la imagen: {e}")

        # Limpiar archivo temporal
        os.remove(tmp_path)
except Exception as e:
    st.error("❌ Error durante la ejecución de la app.")
    st.code(traceback.format_exc())
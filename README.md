# 🎬 Recomendador de Películas por Similitud Visual
Este proyecto implementa un sistema de recomendación de películas basado en la similitud visual de los pósters. Utiliza técnicas de visión por computadora y aprendizaje automático para extraer características visuales y agrupar películas similares. Además, permite al usuario subir una imagen personalizada para obtener recomendaciones basadas en ella.

🚀 Características Principales
Extracción de Características Visuales: Se extraen histogramas de color (RGB y HSV), descriptores HOG, SIFT y momentos de Zernike de los pósters de películas.

Reducción de Dimensionalidad: Se aplican técnicas como PCA, SVD y UMAP para reducir la dimensionalidad de los datos y facilitar la visualización y el clustering.

Clustering: Implementa algoritmos de clustering como K-Means++ y DBSCAN para agrupar películas según similitud visual.

Visualización Interactiva: Ofrece visualizaciones interactivas de los clústeres utilizando UMAP y t-SNE.

Recomendaciones Personalizadas: Permite al usuario ingresar un imdbId o subir una imagen para obtener recomendaciones de películas similares.

Integración con TMDb API: Recupera información y pósters de películas utilizando la API de The Movie Database (TMDb).

🛠️ Tecnologías Utilizadas
Lenguaje de Programación: Python

Librerías Principales:

Streamlit - Para la creación de la interfaz web interactiva.

OpenCV - Para procesamiento de imágenes.

Scikit-learn - Para algoritmos de aprendizaje automático.

UMAP-learn - Para reducción de dimensionalidad.

Seaborn y Matplotlib - Para visualización de datos.

Mahotas - Para cálculo de momentos de Zernike.

Requests - Para realizar solicitudes HTTP a la API de TMDb.



📸 Uso de la Aplicación
Análisis de Componentes Principales: Visualiza la varianza explicada por PCA y SVD.

Clustering: Ajusta los parámetros de K-Means++ y DBSCAN para agrupar películas.

Visualización: Observa los clústeres en 2D utilizando UMAP o t-SNE.

Recomendaciones:

Ingresa un imdbId para obtener películas similares.

Sube una imagen personalizada para recibir recomendaciones basadas en ella.

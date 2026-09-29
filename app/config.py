"""
config.py — 🔧 Configuración del proyecto en un solo lugar. 🔧

Modelos, umbrales y rutas. Los usan la API, la ingesta y la evaluación,
así que un cambio acá se aplica a todos por igual.
"""

import os

# Raíz del proyecto: las rutas no dependen de la carpeta desde donde se corre
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_DATOS = os.path.join(RAIZ, "data")


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

ARCHIVO_LEY = os.path.join(CARPETA_DATOS, "ley_20744.txt")
LEY = "20.744"

CARPETA_CHROMA = os.path.join(CARPETA_DATOS, "chroma_data")
COLECCION = "ley_laboral"

# Se generan solos con el uso
ARCHIVO_CACHE = os.path.join(CARPETA_DATOS, "cache_respuestas.json")
ARCHIVO_REVISIONES = os.path.join(CARPETA_DATOS, "revisiones.json")


# ---------------------------------------------------------------------------
# Modelos de Cohere
# ---------------------------------------------------------------------------

MODELO_EMBEDDINGS = "embed-multilingual-v3.0"
MODELO_CHAT = "command-a-03-2025"
MODELO_RERANK = "rerank-v4.0-fast"


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

# Cuantos artículos se recuperan por similitud antes de reordenar.🔟
# Pido de mas a proposito xq el embedding no siempre deja el artículo
# correcto arriba (por ejemplo: en "período de prueba" entraba el art. 50,
# que trata la prueba del contrato. Con 10 candidatos el reranker
# tiene margen para reordenar. 🔟✅

CANDIDATOS = 10

# Cuantos quedan finalmente como contexto para el modelo.
# Las consultas típicas se responden con 1 (un) artículo, o con 2 que se
# remiten entre si (ej. 92 bis y 231). Más contexto aumenta el riesgo
# de que el modelo cite artículos que no responden la consulta. 😬✋

TOP_K = 3

# Umbral de pertinencia sobre la similitud de embeddings.
# Filtro: decide si la consulta es sobre la ley, antes de
# gastar una llamada al reranker. 💹

UMBRAL_SIMILITUD = 0.45

# Umbral sobre el score del reranker.

UMBRAL_RERANK = 0.30


# ---------------------------------------------------------------------------
# Generación
# ---------------------------------------------------------------------------

# temperature=0 es lo que hace la respuesta reproducible. 🔢✅
TEMPERATURA = 0.0


# ---------------------------------------------------------------------------
# Human in the Loop
# ---------------------------------------------------------------------------

UMBRAL_CONFIANZA = 0.50

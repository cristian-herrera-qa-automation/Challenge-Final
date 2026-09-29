"""
vector_store.py — 📁🔍 Conexión con la base vectorial (ChromaDB). 📁🔍

Este archivo NO carga documentos: eso lo hace scripts/ingesta.py.
Acá solo se consulta la base que ya está en disco.
"""

import chromadb
from chromadb.config import Settings

from app.config import CARPETA_CHROMA, COLECCION

chroma_client = chromadb.PersistentClient(
    path=CARPETA_CHROMA,
    settings=Settings(anonymized_telemetry=False),
)

try:
    coleccion = chroma_client.get_collection(name=COLECCION)
except Exception:
    raise RuntimeError(
        f"No existe la colección '{COLECCION}' en {CARPETA_CHROMA}.\n"
        "Corré primero:  python scripts/ingesta.py"
    )


def contar():
    """Cuántos fragmentos hay indexados."""
    return coleccion.count()


def _distancia_a_similitud(distancia):
    """🔟✅ Chroma devuelve distancia coseno (0 a 2). La pasamos a 0-1.🔟✅"""
    return max(0.0, min(1.0, 1.0 - distancia))


def buscar_por_vector(vector, top_k, excluir_derogados=False):
    """
    🧪✅🔍Busca los articulos más parecidos al vector de la consulta.

    Devuelve una lista de diccionarios con el texto del articulo,
    su metadata y el score de similitud. 🧪✅🔍

    Con excluir_derogados=True se usa la metadata para filtrar en la
    propia búsqueda: los artículos derogados ni siquiera compiten.
    """
    # Pedimos de más porque después descartamos fragmentos repetidos
    # del mismo artículo. 🔍📃
    resultados = coleccion.query(
        query_embeddings=[vector],
        n_results=top_k * 3,
        where={"derogado": False} if excluir_derogados else None,
    )

    if not resultados["ids"] or not resultados["ids"][0]:
        return []

    vistos = {}

    for i in range(len(resultados["ids"][0])):
        meta = resultados["metadatas"][0][i]
        clave = meta["articulo"]

        # Chroma devuelve ordenado de mejor a peor: Entonces nos quedamos
        # con el primer fragmento de cada artículo.📃✅
        if clave in vistos:
            continue

        vistos[clave] = {
            "articulo": meta["articulo"],
            "titulo": meta["titulo"],
            "titulo_nombre": meta["titulo_nombre"],
            "capitulo": meta["capitulo"],
            "capitulo_nombre": meta["capitulo_nombre"],
            "derogado": meta["derogado"],
            "texto": resultados["documents"][0][i],
            "score": round(_distancia_a_similitud(resultados["distances"][0][i]), 4),
        }

    return list(vistos.values())[:top_k]

"""
conftest.py — Preparación de las pruebas.

Las pruebas NO llaman a Cohere ni leen la base vectorial real: las
reemplazamos por versiones falsas que responden lo que cada prueba
necesita. Así:
  - no gastan llamadas de la API (la clave Trial tiene 1000 por mes)
  - dan siempre el mismo resultado (el modelo real no es 100% repetible)
  - corren en cualquier computadora, aunque no se haya hecho la ingesta
  - se pueden provocar casos difíciles de conseguir con el modelo real:
    Cohere caído, reranker caído, salida degenerada

Lo que se prueba es NUESTRO código: filtros, umbrales, Human in the Loop,
caché y manejo de errores.

Para correr además la prueba real contra Cohere (gasta ~3 llamadas):
    PRUEBA_REAL=1 pytest -m real          (bash)
    $env:PRUEBA_REAL=1; pytest -m real    (PowerShell)
"""

import os
import sys
from collections import Counter
from types import SimpleNamespace

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
os.chdir(RAIZ)

PRUEBA_REAL = os.getenv("PRUEBA_REAL") == "1"


# ---------------------------------------------------------------------------
# Escenario: lo que "contestan" Cohere y Chroma en cada prueba
# ---------------------------------------------------------------------------

class Escenario:
    def __init__(self):
        self.reiniciar()

    def reiniciar(self):
        self.candidatos = []          # artículos que devuelve la búsqueda
        self.scores_rerank = {}       # articulo -> score del reranker
        self.rerank_caido = False
        self.embed_caido = False
        self.respuestas_chat = []     # lo que "genera" el modelo, en orden
        self.llamadas = Counter()     # cuántas veces se llamó a cada cosa
        self.ultimo_where = None      # filtro de metadata usado en la búsqueda


escenario = Escenario()


def articulo(numero, similitud, derogado=False, texto=None):
    """Arma un artículo falso para la base vectorial."""
    return {
        "articulo": numero,
        "similitud": similitud,
        "derogado": derogado,
        "texto": texto or f"Art. {numero}. — Texto de prueba del artículo {numero}.",
    }


class CohereFalso:
    def embed(self, **kwargs):
        escenario.llamadas["embed"] += 1
        if escenario.embed_caido:
            raise ConnectionError("Cohere no responde")
        return SimpleNamespace(embeddings=SimpleNamespace(float=[[0.1, 0.2, 0.3]]))

    def rerank(self, query, documents, top_n, **kwargs):
        escenario.llamadas["rerank"] += 1
        if escenario.rerank_caido:
            raise ConnectionError("Rerank no responde")
        por_texto = {c["texto"]: c["articulo"] for c in escenario.candidatos}
        puntajes = [(i, escenario.scores_rerank.get(por_texto[doc], 0.0))
                    for i, doc in enumerate(documents)]
        puntajes.sort(key=lambda p: p[1], reverse=True)
        return SimpleNamespace(results=[
            SimpleNamespace(index=i, relevance_score=s) for i, s in puntajes[:top_n]
        ])

    def chat(self, **kwargs):
        escenario.llamadas["chat"] += 1
        # Va consumiendo las respuestas en orden; la última se repite.
        texto = (escenario.respuestas_chat.pop(0) if len(escenario.respuestas_chat) > 1
                 else escenario.respuestas_chat[0])
        return SimpleNamespace(message=SimpleNamespace(content=[SimpleNamespace(text=texto)]))


class ColeccionFalsa:
    def count(self):
        return 317

    def query(self, query_embeddings, n_results, where=None):
        escenario.ultimo_where = where
        candidatos = escenario.candidatos
        if where == {"derogado": False}:
            candidatos = [c for c in candidatos if not c["derogado"]]
        candidatos = sorted(candidatos, key=lambda c: c["similitud"], reverse=True)[:n_results]
        return {
            "ids": [[f"id_{c['articulo']}" for c in candidatos]],
            "documents": [[c["texto"] for c in candidatos]],
            "distances": [[1 - c["similitud"] for c in candidatos]],
            "metadatas": [[{
                "articulo": c["articulo"],
                "titulo": "I", "titulo_nombre": "Título de prueba",
                "capitulo": "I", "capitulo_nombre": "Capítulo de prueba",
                "derogado": c["derogado"],
            } for c in candidatos]],
        }


# ---------------------------------------------------------------------------
# Reemplazo ANTES de importar la app (rag.py se conecta al importarse)
# ---------------------------------------------------------------------------

if not PRUEBA_REAL:
    import cohere
    import chromadb

    os.environ.setdefault("COHERE_API_KEY", "clave-falsa-para-pruebas")
    cohere.ClientV2 = lambda **kwargs: CohereFalso()
    chromadb.PersistentClient = lambda **kwargs: SimpleNamespace(
        get_collection=lambda name: ColeccionFalsa()
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "real: prueba contra Cohere real (gasta llamadas)")


def pytest_collection_modifyitems(config, items):
    for item in items:
        es_real = "real" in item.keywords
        if es_real and not PRUEBA_REAL:
            item.add_marker(pytest.mark.skip(reason="Prueba real: correr con PRUEBA_REAL=1"))
        if not es_real and PRUEBA_REAL:
            item.add_marker(pytest.mark.skip(reason="Con PRUEBA_REAL=1 solo corren las pruebas reales"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def estado(tmp_path, monkeypatch):
    """
    Deja todo limpio para cada prueba: escenario vacío, y caché y
    revisiones en una carpeta temporal (no se tocan los archivos reales).
    """
    import main
    import revisiones

    escenario.reiniciar()
    monkeypatch.setattr(main, "cache", {})
    monkeypatch.setattr(main, "ARCHIVO_CACHE", str(tmp_path / "cache.json"))
    monkeypatch.setattr(revisiones, "revisiones", {})
    monkeypatch.setattr(revisiones, "ARCHIVO", str(tmp_path / "revisiones.json"))
    return escenario


@pytest.fixture
def cliente(estado):
    from fastapi.testclient import TestClient
    import main
    return TestClient(main.app)

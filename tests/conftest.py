"""
conftest.py — Preparación de las pruebas. 🧪👨‍💻

Los test usan mi sistema: El LLM y la base vectorial
de verdad (data/chroma_data).

Lo único que cambia es dónde se guardan el caché y las revisiones: van a
una carpeta temporal, así cada corrida llama al modelo de verdad y no se
tocan los archivos de data/.

Se necesita el .env con la clave de Cohere y haber corrido la ingesta.
Si falta algo, las pruebas se saltean con un aviso en vez de fallar.
🧪👨‍💻
"""

import os
import sys

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
os.chdir(RAIZ)

from dotenv import load_dotenv

load_dotenv(os.path.join(RAIZ, ".env"))


@pytest.fixture(scope="session")
def cliente(tmp_path_factory):
    """La API real, con caché y revisiones en una carpeta temporal."""
    if not os.getenv("COHERE_API_KEY"):
        pytest.skip("Falta COHERE_API_KEY en el archivo .env")

    try:
        import main
    except RuntimeError as e:
        # vector_store.py avisa si no se hizo la ingesta
        pytest.skip(str(e))

    from fastapi.testclient import TestClient
    from app.services import cache_service, revision_service

    carpeta = tmp_path_factory.mktemp("datos_de_prueba")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cache_service, "respuestas", {})
        mp.setattr(cache_service, "ARCHIVO", str(carpeta / "cache.json"))
        mp.setattr(revision_service, "revisiones", {})
        mp.setattr(revision_service, "ARCHIVO", str(carpeta / "revisiones.json"))
        yield TestClient(main.app)

"""
test_real.py — Una prueba de punta a punta contra Cohere y la base real.

No corre por defecto porque gasta llamadas de la API (~3 por prueba).
Necesita el .env con la clave y haber corrido la ingesta.

    PRUEBA_REAL=1 pytest -m real -v          (bash)
    $env:PRUEBA_REAL=1; pytest -m real -v    (PowerShell)
"""

import pytest


@pytest.mark.real
def test_vacaciones_contra_cohere_real(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import main

    # Caché en una carpeta temporal: queremos que llame al modelo de verdad
    monkeypatch.setattr(main, "cache", {})
    monkeypatch.setattr(main, "ARCHIVO_CACHE", str(tmp_path / "cache.json"))
    cliente = TestClient(main.app)

    r = cliente.post("/ask", json={
        "pregunta": "¿Cuántos días de vacaciones me corresponden con 8 años de antigüedad?"
    })

    assert r.status_code == 200
    datos = r.json()
    assert datos["estado"] == "respondida"
    assert datos["fuentes"][0]["articulo"] == "150"
    assert "21" in datos["respuesta"]

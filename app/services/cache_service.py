"""
cache_service.py — Caché de respuestas. 📁✅

Garantiza que la misma pregunta devuelva la misma respuesta, y ahorra
llamadas a Cohere. Se guarda en un archivo JSON para que sobreviva a un
reinicio de la API.
"""

import re
import json
import logging
import unicodedata

from app.config import ARCHIVO_CACHE

logger = logging.getLogger("asistente_laboral")

ARCHIVO = ARCHIVO_CACHE


def _cargar():
    try:
        with open(ARCHIVO, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


respuestas = _cargar()


def _guardar_en_disco():
    try:
        with open(ARCHIVO, "w", encoding="utf-8") as f:
            json.dump(respuestas, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("No se pudo guardar el cache: %s", type(e).__name__)


def clave(pregunta):
    """
    Normaliza la pregunta para usarla como clave.

    "¿Cuántos días de vacaciones?" y "cuantos dias de vacaciones"
    son la misma consulta, así que deben devolver la misma respuesta.
    Se pasa a minúsculas, se sacan tildes y signos, y se colapsan espacios.
    """
    texto = pregunta.lower().strip()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^\w\s]", "", texto)
    return re.sub(r"\s+", " ", texto).strip()


def obtener(clave_pregunta):
    """La respuesta guardada para esta clave, o None."""
    return respuestas.get(clave_pregunta)


def guardar(clave_pregunta, respuesta):
    respuestas[clave_pregunta] = respuesta
    _guardar_en_disco()


def cantidad():
    return len(respuestas)

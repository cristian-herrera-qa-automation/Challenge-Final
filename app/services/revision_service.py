"""
revision_service.py — Human in the Loop: respuestas que un humano debe aprobar. 👨‍💻✋

La IA responde sola cuando la evidencia es clara. ✅ Cuando no lo es, la
respuesta se genera pero NO se entrega: ✋ queda en estado pendiente hasta
que un revisor la aprueba o la rechaza.

    IA analiza -> evalúa si requiere supervisión -> revisión humana
               -> aprobar (se entrega) 👍 o rechazar (se detiene) 🚨


Las revisiones se guardan en un archivo JSON, igual que el caché, para
que sobrevivan a un reinicio de la API.
"""

import json
import uuid
import logging
from datetime import datetime

from app.config import UMBRAL_CONFIANZA, ARCHIVO_REVISIONES

logger = logging.getLogger("asistente_laboral")


# ---------------------------------------------------------------------------
# Criterios de riesgo
# ---------------------------------------------------------------------------

MOTIVO_DEROGADO = "cita_articulo_derogado"
MOTIVO_CONFIANZA_BAJA = "confianza_baja"
MOTIVO_SIN_RERANK = "confianza_no_medida"


def evaluar_riesgo(fuentes):
    """
    Devuelve la lista de motivos por los que la respuesta necesita
    revisión humana. 📃👨‍💻 Lista vacía = se puede entregar automáticamente.

    Recibe las fuentes finales, las que pasaron los dos filtros: un
    artículo derogado que quedó descartado no afecta la respuesta. 📃
    """
    motivos = []

    if any(f["derogado"] for f in fuentes):
        motivos.append(MOTIVO_DEROGADO)

    scores = [f["score_rerank"] for f in fuentes if f.get("score_rerank") is not None]
    if not scores:
        # El reranker no respondió: no hay medida de confianza.
        motivos.append(MOTIVO_SIN_RERANK)
    elif max(scores) < UMBRAL_CONFIANZA:
        motivos.append(MOTIVO_CONFIANZA_BAJA)

    return motivos


# ---------------------------------------------------------------------------
# Almacenamiento
# ---------------------------------------------------------------------------

ARCHIVO = ARCHIVO_REVISIONES

PENDIENTE = "pendiente"
APROBADA = "aprobada"
RECHAZADA = "rechazada"


def _cargar():
    try:
        with open(ARCHIVO, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


revisiones = _cargar()


def _guardar():
    try:
        with open(ARCHIVO, "w", encoding="utf-8") as f:
            json.dump(revisiones, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("No se pudieron guardar las revisiones: %s", type(e).__name__)


def _ahora():
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Operaciones
# ---------------------------------------------------------------------------

def crear(clave, pregunta, propuesta, motivos):
    """🧪🎯Registra una respuesta que espera revisión y devuelve su id.🧪🎯"""
    id_revision = uuid.uuid4().hex[:8]
    revisiones[id_revision] = {
        "id": id_revision,
        "estado": PENDIENTE,
        "clave": clave,
        "pregunta": pregunta,
        "motivos": motivos,
        "propuesta": propuesta,
        "creada": _ahora(),
        "revisor": None,
        "comentario": None,
        "resuelta": None,
    }
    _guardar()
    return id_revision


def obtener(id_revision):
    return revisiones.get(id_revision)


def buscar_por_clave(clave):
    """
    La revisión más reciente de esta misma pregunta, si existe.

    Evita que preguntar lo mismo dos veces genere dos revisiones, y que
    una respuesta rechazada se vuelva a generar y se entregue sola.🔑📁
    """
    candidatas = [r for r in revisiones.values() if r["clave"] == clave]
    return max(candidatas, key=lambda r: r["creada"], default=None)


def pendientes():
    return [r for r in revisiones.values() if r["estado"] == PENDIENTE]


def resolver(id_revision, aprobada, revisor, comentario=None):
    """🚨✅👨‍💻Aprueba o rechaza. Devuelve la revisión actualizada.🚨✅👨‍💻"""
    
    revision = revisiones[id_revision]
    revision["estado"] = APROBADA if aprobada else RECHAZADA
    revision["revisor"] = revisor
    revision["comentario"] = comentario
    revision["resuelta"] = _ahora()
    _guardar()
    return revision

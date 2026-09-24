"""
main.py — La API del asistente sobre la Ley de Contrato de Trabajo.

Levantar con:  uvicorn main:app --reload
Documentación: http://localhost:8000/docs
"""

import re
import json
import uuid
import logging
import unicodedata

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()  # antes de importar rag: ahí se lee la API key

import rag
from schemas import ConsultaRequest, ConsultaResponse


# ---------------------------------------------------------------------------
# Logging: cada paso de cada consulta queda registrado
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("asistente_laboral")


# ---------------------------------------------------------------------------
# Mensajes fijos
# ---------------------------------------------------------------------------

CONSULTA_BLOQUEADA = "No puedo responder a este tipo de consultas."
FUERA_DE_TEMA = (
    "Esta consulta no corresponde a la Ley de Contrato de Trabajo. "
    "Solo puedo responder sobre el contenido de la Ley 20.744."
)
SERVICIO_CAIDO = "El servicio externo no pudo procesar la solicitud en este momento."

AVISO_LEGAL = (
    "Información general basada en el texto de la Ley 20.744. "
    "No constituye asesoramiento legal. Ante un caso concreto, "
    "consultá a un profesional del derecho."
)


app = FastAPI(
    title="Asistente de Consulta - Ley de Contrato de Trabajo",
    description="Sistema RAG sobre la Ley 20.744 (Argentina)",
    version="1.0",
)


@app.exception_handler(HTTPException)
async def formato_de_error(request: Request, exc: HTTPException):
    """Todos los errores salen como {"error": "..."} y sin detalles internos."""
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


# ---------------------------------------------------------------------------
# Caché: garantiza que la misma pregunta devuelva la misma respuesta
# ---------------------------------------------------------------------------

ARCHIVO_CACHE = "cache_respuestas.json"


def _cargar_cache():
    try:
        with open(ARCHIVO_CACHE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


cache = _cargar_cache()


def _guardar_cache():
    try:
        with open(ARCHIVO_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("No se pudo guardar el cache: %s", type(e).__name__)


def clave_cache(pregunta):
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


# ---------------------------------------------------------------------------
# POST /consultar
# ---------------------------------------------------------------------------

@app.post("/consultar", response_model=ConsultaResponse)
async def consultar(req: ConsultaRequest):
    traza = uuid.uuid4().hex[:8]
    logger.info("PASO 1 CONSULTA | traza=%s | largo=%d", traza, len(req.pregunta))

    # --- Guardrail de lenguaje (no se busca ni se llama al modelo) ---
    if rag.contiene_lenguaje_inapropiado(req.pregunta):
        logger.warning("PASO 2 GUARDRAIL | traza=%s | consulta bloqueada", traza)
        return _respuesta_simple(req.pregunta, CONSULTA_BLOQUEADA)

    # --- Caché: si ya respondimos esta pregunta, devolvemos lo mismo ---
    clave = clave_cache(req.pregunta)
    if clave in cache:
        logger.info("PASO 2 CACHE | traza=%s | respuesta reutilizada", traza)
        guardada = dict(cache[clave])
        guardada["pregunta"] = req.pregunta
        guardada["desde_cache"] = True
        return ConsultaResponse(**guardada)

    # --- Búsqueda en la base vectorial ---
    try:
        articulos = rag.buscar(req.pregunta)
    except Exception as e:
        logger.error("PASO 3 BUSQUEDA | traza=%s | fallo: %s", traza, type(e).__name__)
        raise HTTPException(status_code=503, detail=SERVICIO_CAIDO)

    if not articulos:
        logger.info("PASO 3 BUSQUEDA | traza=%s | sin resultados", traza)
        return _respuesta_simple(req.pregunta, FUERA_DE_TEMA)

    mejor = articulos[0]
    logger.info(
        "PASO 3 BUSQUEDA | traza=%s | %d candidatos | mejor=%s score=%.4f",
        traza, len(articulos), mejor["articulo"], mejor["score"],
    )

    # --- Filtro 1 (barato): ¿la consulta es sobre la ley? ---
    # Si no supera el umbral de similitud, cortamos acá y nos ahorramos
    # la llamada al reranker y al modelo.
    if mejor["score"] < rag.UMBRAL_SIMILITUD:
        logger.info(
            "PASO 4 PERTINENCIA | traza=%s | score=%.4f < umbral=%.2f | fuera de tema",
            traza, mejor["score"], rag.UMBRAL_SIMILITUD,
        )
        return _respuesta_simple(req.pregunta, FUERA_DE_TEMA, mejor["score"])

    # --- Reranking: de los candidatos, cuáles responden REALMENTE ---
    try:
        articulos = rag.reordenar(req.pregunta, articulos)
    except Exception as e:
        logger.error("PASO 4 RERANK | traza=%s | fallo: %s", traza, type(e).__name__)
        raise HTTPException(status_code=503, detail=SERVICIO_CAIDO)

    # --- Filtro 2: descartar los que el reranker considera irrelevantes ---
    articulos = [
        a for a in articulos
        if a.get("score_rerank", 1.0) >= rag.UMBRAL_RERANK
    ]

    if not articulos:
        logger.info("PASO 4 RERANK | traza=%s | ningun articulo relevante", traza)
        return _respuesta_simple(req.pregunta, FUERA_DE_TEMA, mejor["score"])

    logger.info(
        "PASO 4 RERANK | traza=%s | %s",
        traza,
        [(a["articulo"], a.get("score_rerank")) for a in articulos],
    )

    # --- Generación ---
    contexto = rag.armar_contexto(articulos)
    try:
        texto = rag.generar_respuesta(req.pregunta, contexto)
    except Exception as e:
        logger.error("PASO 5 GENERACION | traza=%s | fallo: %s", traza, type(e).__name__)
        raise HTTPException(status_code=503, detail=SERVICIO_CAIDO)

    texto = rag.quitar_emojis(texto)

    # El modelo puede decidir por su cuenta que el contexto no alcanza.
    # Si lo dijo, grounded va en False aunque el score haya pasado.
    fundamentada = rag.SIN_CONTEXTO.lower()[:40] not in texto.lower()

    respuesta = ConsultaResponse(
        pregunta=req.pregunta,
        respuesta=texto,
        articulos=[{
            "articulo": a["articulo"],
            "titulo": f"{a['titulo']} - {a['titulo_nombre']}".strip(" -"),
            "capitulo": f"{a['capitulo']} - {a['capitulo_nombre']}".strip(" -"),
            "derogado": a["derogado"],
            "score": a["score"],
            "score_rerank": a.get("score_rerank"),
        } for a in articulos],
        similarity_score=mejor["score"],
        grounded=fundamentada,
        desde_cache=False,
        aviso=AVISO_LEGAL,
    )

    # Guardamos para que la próxima vez la respuesta sea idéntica
    cache[clave] = respuesta.model_dump()
    _guardar_cache()

    logger.info(
        "PASO 5 GENERACION | traza=%s | articulos=%s | grounded=%s",
        traza, [a["articulo"] for a in articulos], fundamentada,
    )

    return respuesta


# ---------------------------------------------------------------------------
# Otros endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def raiz():
    return {
        "servicio": "Asistente de Consulta - Ley de Contrato de Trabajo",
        "ley": "20.744",
        "endpoint": "/consultar",
        "documentacion": "/docs",
    }


@app.get("/estado")
async def estado():
    """Verifica que la base vectorial esté cargada. Útil antes de una demo."""
    return {
        "fragmentos_indexados": rag.coleccion.count(),
        "modelo_embeddings": rag.MODELO_EMBEDDINGS,
        "modelo_chat": rag.MODELO_CHAT,
        "modelo_rerank": rag.MODELO_RERANK,
        "umbral_similitud": rag.UMBRAL_SIMILITUD,
        "umbral_rerank": rag.UMBRAL_RERANK,
        "respuestas_en_cache": len(cache),
    }


# ---------------------------------------------------------------------------
# Auxiliar
# ---------------------------------------------------------------------------

def _respuesta_simple(pregunta, texto, score=0.0):
    """Respuesta sin artículos: bloqueada o fuera de tema."""
    return ConsultaResponse(
        pregunta=pregunta,
        respuesta=texto,
        articulos=[],
        similarity_score=score,
        grounded=False,
        desde_cache=False,
        aviso=AVISO_LEGAL,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
main.py — La API del asistente sobre la Ley de Contrato de Trabajo.

Levantar con:  uvicorn main:app --reload
Documentación: http://localhost:8000/docs

Endpoints:
    GET  /health     estado del servicio y de la base vectorial
    POST /retrieve   solo retrieval: qué fragmentos se recuperan y por qué
    POST /ask        pregunta -> retrieval -> contexto -> prompt -> LLM -> respuesta + fuentes
"""

import re
import json
import uuid
import logging
import unicodedata

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from dotenv import load_dotenv

load_dotenv()  # antes de importar rag: ahí se lee la API key

import rag
from schemas import (
    PreguntaRequest, RetrieveRequest, RetrieveResponse, AskResponse, ErrorResponse,
)


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

ERRORES = {503: {"model": ErrorResponse, "description": "Cohere no respondió"}}


app = FastAPI(
    title="Asistente de Consulta - Ley de Contrato de Trabajo",
    description="Sistema RAG sobre la Ley 20.744 (Argentina)",
    version="2.0",
)


# Se registra sobre la excepción de Starlette (la base de la de FastAPI)
# para cubrir también los errores que arma el framework: 404, cuerpo
# mal formado, etc.
@app.exception_handler(StarletteHTTPException)
async def formato_de_error(request: Request, exc: StarletteHTTPException):
    """Todos los errores salen como {"error": "..."} y sin detalles internos."""
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(RequestValidationError)
async def error_de_validacion(request: Request, exc: RequestValidationError):
    """Una entrada inválida (ej. pregunta vacía) sale con el mismo formato."""
    primero = exc.errors()[0]
    # loc es ("body", "pregunta") para un campo, o ("body", 13) cuando el
    # JSON está roto (13 = posición del error): solo nombramos campos.
    campo = ".".join(p for p in primero["loc"] if isinstance(p, str) and p != "body")
    mensaje = primero["msg"].removeprefix("Value error, ")
    return JSONResponse(
        status_code=422,
        content={"error": f"{campo}: {mensaje}" if campo else mensaje},
    )


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
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Verifica que la base vectorial esté cargada. Útil antes de una demo."""
    try:
        fragmentos = rag.coleccion.count()
    except Exception as e:
        logger.error("HEALTH | base vectorial no disponible: %s", type(e).__name__)
        fragmentos = 0

    return {
        "status": "ok" if fragmentos > 0 else "degradado",
        "fragmentos_indexados": fragmentos,
        "modelo_embeddings": rag.MODELO_EMBEDDINGS,
        "modelo_chat": rag.MODELO_CHAT,
        "modelo_rerank": rag.MODELO_RERANK,
        "top_k": rag.TOP_K,
        "umbral_similitud": rag.UMBRAL_SIMILITUD,
        "umbral_rerank": rag.UMBRAL_RERANK,
        "respuestas_en_cache": len(cache),
    }


# ---------------------------------------------------------------------------
# POST /retrieve — solo retrieval, para inspeccionarlo sin generar
# ---------------------------------------------------------------------------

@app.post("/retrieve", response_model=RetrieveResponse, responses=ERRORES)
async def retrieve(req: RetrieveRequest):
    traza = uuid.uuid4().hex[:8]
    logger.info("PASO 1 RETRIEVE | traza=%s | largo=%d", traza, len(req.pregunta))

    try:
        resultado = rag.recuperar(
            req.pregunta,
            top_k=req.top_k,
            excluir_derogados=req.excluir_derogados,
            traza=traza,
        )
    except Exception as e:
        logger.error("PASO 3 BUSQUEDA | traza=%s | fallo: %s", traza, type(e).__name__)
        raise HTTPException(status_code=503, detail=SERVICIO_CAIDO)

    return RetrieveResponse(
        pregunta=req.pregunta,
        pertinente=resultado["pertinente"],
        motivo=resultado["motivo"],
        similarity_score=resultado["similarity_score"],
        umbral_similitud=rag.UMBRAL_SIMILITUD,
        umbral_rerank=rag.UMBRAL_RERANK,
        fragmentos=[{**_fuente(a), "texto": a["texto"]}
                    for a in resultado["fragmentos"]],
    )


# ---------------------------------------------------------------------------
# POST /ask — pregunta completa: retrieval + generación
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse, responses=ERRORES)
async def ask(req: PreguntaRequest):
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
        return AskResponse(**guardada)

    # --- Retrieval: búsqueda + filtro de similitud + reranking ---
    try:
        resultado = rag.recuperar(req.pregunta, traza=traza)
    except Exception as e:
        logger.error("PASO 3 BUSQUEDA | traza=%s | fallo: %s", traza, type(e).__name__)
        raise HTTPException(status_code=503, detail=SERVICIO_CAIDO)

    if not resultado["pertinente"]:
        return _respuesta_simple(req.pregunta, FUERA_DE_TEMA,
                                 resultado["similarity_score"])

    articulos = resultado["fragmentos"]

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

    respuesta = AskResponse(
        pregunta=req.pregunta,
        respuesta=texto,
        fuentes=[_fuente(a) for a in articulos],
        similarity_score=resultado["similarity_score"],
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
# Raíz
# ---------------------------------------------------------------------------

@app.get("/")
async def raiz():
    return {
        "servicio": "Asistente de Consulta - Ley de Contrato de Trabajo",
        "ley": "20.744",
        "endpoints": ["GET /health", "POST /retrieve", "POST /ask"],
        "documentacion": "/docs",
    }


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------

def _fuente(a):
    """Pasa un artículo recuperado al formato que devuelve la API."""
    return {
        "articulo": a["articulo"],
        "titulo": f"{a['titulo']} - {a['titulo_nombre']}".strip(" -"),
        "capitulo": f"{a['capitulo']} - {a['capitulo_nombre']}".strip(" -"),
        "derogado": a["derogado"],
        "score": a["score"],
        "score_rerank": a.get("score_rerank"),
    }


def _respuesta_simple(pregunta, texto, score=0.0):
    """Respuesta sin fuentes: bloqueada o fuera de tema."""
    return AskResponse(
        pregunta=pregunta,
        respuesta=texto,
        fuentes=[],
        similarity_score=score,
        grounded=False,
        desde_cache=False,
        aviso=AVISO_LEGAL,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

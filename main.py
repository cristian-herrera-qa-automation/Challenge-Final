"""
main.py — La API del asistente sobre la Ley de Contrato de Trabajo.

Levantar con:  uvicorn main:app --reload
Documentación: http://localhost:8000/docs

Endpoints:
    GET  /health     estado del servicio y de la base vectorial
    POST /retrieve   solo retrieval: qué fragmentos se recuperan y por qué
    POST /ask        pregunta -> retrieval -> contexto -> prompt -> LLM -> respuesta + fuentes

Human in the Loop:
    GET  /revisiones/pendientes        el revisor ve qué espera aprobación
    GET  /revisiones/{id}              el usuario consulta cómo terminó su pregunta
    POST /revisiones/{id}/aprobar      el revisor aprueba: la respuesta se entrega
    POST /revisiones/{id}/rechazar     el revisor rechaza: la respuesta no sale
"""

import re
import json
import uuid
import logging
import unicodedata

from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from dotenv import load_dotenv

load_dotenv()  # antes de importar rag: ahí se lee la API key

import rag
import revisiones
from schemas import (
    PreguntaRequest, RetrieveRequest, RetrieveResponse, AskResponse, ErrorResponse,
    ResolucionRequest, RevisionPendiente, EstadoRevision,
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

# Mensaje para la persona que pregunta. Los datos técnicos (id y motivos)
# viajan en sus propios campos: id_revision y motivos_revision.
EN_REVISION = (
    "Tu consulta necesita la revisión de una persona antes de responderse, "
    "porque la evidencia encontrada requiere verificación."
)
RECHAZADA = (
    "Un revisor no pudo confirmar una respuesta confiable para esta consulta "
    "en la Ley 20.744. Te recomendamos consultar a un profesional del derecho."
)

ERRORES = {503: {"model": ErrorResponse, "description": "Cohere no respondió"}}
NO_EXISTE = {404: {"model": ErrorResponse, "description": "No existe esa revisión"}}


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
        "revisiones_pendientes": len(revisiones.pendientes()),
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

    # --- ¿Esta misma pregunta ya pasó por revisión humana? ---
    # Pendiente: no generamos otra. Rechazada: respetamos la decisión.
    # (Las aprobadas ya están en el caché.)
    previa = revisiones.buscar_por_clave(clave)
    if previa and previa["estado"] == revisiones.PENDIENTE:
        logger.info("PASO 2 HITL | traza=%s | sigue pendiente id=%s", traza, previa["id"])
        return _respuesta_en_revision(req.pregunta, previa)
    if previa and previa["estado"] == revisiones.RECHAZADA:
        logger.info("PASO 2 HITL | traza=%s | rechazada antes id=%s", traza, previa["id"])
        return _respuesta_rechazada(req.pregunta, previa)

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

    # --- Human in the Loop: ¿se puede entregar sin que la mire nadie? ---
    # Si el modelo ya dijo que no tiene información, no hay nada que
    # revisar: se está negando a responder.
    motivos = revisiones.evaluar_riesgo(respuesta.model_dump()["fuentes"]) if fundamentada else []
    if motivos:
        id_revision = revisiones.crear(clave, req.pregunta, respuesta.model_dump(), motivos)
        logger.warning(
            "PASO 6 HITL | traza=%s | pending_approval id=%s | motivos=%s",
            traza, id_revision, motivos,
        )
        return _respuesta_en_revision(req.pregunta, revisiones.obtener(id_revision))

    # Guardamos para que la próxima vez la respuesta sea idéntica
    cache[clave] = respuesta.model_dump()
    _guardar_cache()

    logger.info(
        "PASO 5 GENERACION | traza=%s | articulos=%s | grounded=%s",
        traza, [a["articulo"] for a in articulos], fundamentada,
    )

    return respuesta


# ---------------------------------------------------------------------------
# Human in the Loop: revisión de respuestas
# ---------------------------------------------------------------------------

# Va antes que /revisiones/{id_revision}: si no, "pendientes" se tomaría
# como un id.
@app.get("/revisiones/pendientes", response_model=List[RevisionPendiente])
async def revisiones_pendientes():
    """Para el revisor: las respuestas que esperan aprobación, con la propuesta de la IA."""
    return [RevisionPendiente(**r) for r in revisiones.pendientes()]


@app.get("/revisiones/{id_revision}", response_model=EstadoRevision, responses=NO_EXISTE)
async def estado_revision(id_revision: str):
    """Para el usuario: cómo terminó su consulta. La respuesta aparece solo si fue aprobada."""
    return _estado_para_usuario(_revision_o_404(id_revision))


YA_RESUELTA = {409: {"model": ErrorResponse, "description": "Ya fue resuelta"}}


@app.post("/revisiones/{id_revision}/aprobar", response_model=EstadoRevision,
          responses={**NO_EXISTE, **YA_RESUELTA})
async def aprobar(id_revision: str, req: ResolucionRequest):
    """El revisor aprueba: la respuesta se entrega y queda en el caché."""
    revision = _resolver(id_revision, True, req)

    # Desde ahora, la misma pregunta se responde directo con lo aprobado
    cache[revision["clave"]] = revision["propuesta"]
    _guardar_cache()

    return _estado_para_usuario(revision)


@app.post("/revisiones/{id_revision}/rechazar", response_model=EstadoRevision,
          responses={**NO_EXISTE, **YA_RESUELTA})
async def rechazar(id_revision: str, req: ResolucionRequest):
    """El revisor rechaza: la respuesta generada no se entrega nunca."""
    return _estado_para_usuario(_resolver(id_revision, False, req))


def _revision_o_404(id_revision):
    revision = revisiones.obtener(id_revision)
    if not revision:
        raise HTTPException(status_code=404, detail="No existe esa revisión")
    return revision


def _resolver(id_revision, aprobada, req):
    revision = _revision_o_404(id_revision)
    if revision["estado"] != revisiones.PENDIENTE:
        raise HTTPException(status_code=409, detail=f"La revisión ya fue {revision['estado']}")

    revision = revisiones.resolver(id_revision, aprobada, req.revisor, req.comentario)
    logger.info("HITL | revision=%s | %s por revisor", id_revision, revision["estado"])
    return revision


def _estado_para_usuario(revision):
    return EstadoRevision(
        **{k: revision[k] for k in ("id", "estado", "pregunta", "motivos",
                                    "revisor", "comentario", "creada", "resuelta")},
        respuesta=revision["propuesta"] if revision["estado"] == revisiones.APROBADA else None,
    )


# ---------------------------------------------------------------------------
# Raíz
# ---------------------------------------------------------------------------

@app.get("/")
async def raiz():
    return {
        "servicio": "Asistente de Consulta - Ley de Contrato de Trabajo",
        "ley": "20.744",
        "endpoints": ["GET /health", "POST /retrieve", "POST /ask",
                      "GET /revisiones/pendientes", "GET /revisiones/{id}",
                      "POST /revisiones/{id}/aprobar", "POST /revisiones/{id}/rechazar"],
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


def _respuesta_en_revision(pregunta, revision):
    """La respuesta generada queda retenida: solo se informa cómo seguirla."""
    propuesta = revision["propuesta"]
    return AskResponse(
        pregunta=pregunta,
        estado="pending_approval",
        id_revision=revision["id"],
        motivos_revision=revision["motivos"],
        respuesta=EN_REVISION,
        fuentes=propuesta["fuentes"],
        similarity_score=propuesta["similarity_score"],
        grounded=False,
        aviso=AVISO_LEGAL,
    )


def _respuesta_rechazada(pregunta, revision):
    return AskResponse(
        pregunta=pregunta,
        estado="rechazada",
        id_revision=revision["id"],
        motivos_revision=revision["motivos"],
        respuesta=RECHAZADA,
        similarity_score=revision["propuesta"]["similarity_score"],
        grounded=False,
        aviso=AVISO_LEGAL,
    )


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

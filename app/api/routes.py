"""
routes.py — 🌐 Los endpoints de la API. 🌐

    GET  /health     estado del servicio y de la base vectorial (✅)
    POST /retrieve   solo retrieval: qué fragmentos se recuperan y porqué (✅)
    POST /ask        pregunta -> retrieval -> contexto -> prompt -> LLM -> respuesta + fuentes (✅)

Human in the Loop:
    GET  /revisiones/pendientes        la persona ve qué consultas esperan aprobación (✅)
    GET  /revisiones/{id}              el usuario consulta como terminó su pregunta (✅)
    POST /revisiones/{id}/aprobar      la persona aprueba: la respuesta se entrega al usuario (✅)
    POST /revisiones/{id}/rechazar     el revisor rechaza: la respuesta no sale (✅)

Acá solo se recibe el pedido, se orquestan los servicios y se arma la
respuesta. La lógica del RAG está en app/services/.
"""

import uuid
import logging

from typing import List

from fastapi import APIRouter, HTTPException

from app import config
from app.infrastructure import vector_store
from app.services import rag_service, revision_service, cache_service
from app.schemas.rag import (
    PreguntaRequest, RetrieveRequest, RetrieveResponse, AskResponse, ErrorResponse,
)
from app.schemas.revisiones import ResolucionRequest, RevisionPendiente, EstadoRevision

logger = logging.getLogger("asistente_laboral")

router = APIRouter()


# ---------------------------------------------------------------------------
#✍ Mensajes fijos ✍
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
# viajan en sus propios campos: id_revision y motivos_revision. ✍🌐
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


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get("/health")
async def health():
    """Verifica que la base vectorial esté cargada. Útil antes de una demo."""
    try:
        fragmentos = vector_store.contar()
    except Exception as e:
        logger.error("HEALTH | base vectorial no disponible: %s", type(e).__name__)
        fragmentos = 0

    return {
        "status": "ok" if fragmentos > 0 else "degradado",
        "fragmentos_indexados": fragmentos,
        "modelo_embeddings": config.MODELO_EMBEDDINGS,
        "modelo_chat": config.MODELO_CHAT,
        "modelo_rerank": config.MODELO_RERANK,
        "top_k": config.TOP_K,
        "umbral_similitud": config.UMBRAL_SIMILITUD,
        "umbral_rerank": config.UMBRAL_RERANK,
        "respuestas_en_cache": cache_service.cantidad(),
        "revisiones_pendientes": len(revision_service.pendientes()),
    }


# ---------------------------------------------------------------------------
# POST /retrieve — solo retrieval, para inspeccionarlo sin generar
# ---------------------------------------------------------------------------

@router.post("/retrieve", response_model=RetrieveResponse, responses=ERRORES)
async def retrieve(req: RetrieveRequest):
    traza = uuid.uuid4().hex[:8]
    logger.info("PASO 1 RETRIEVE | traza=%s | largo=%d", traza, len(req.pregunta))

    try:
        resultado = rag_service.recuperar(
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
        umbral_similitud=config.UMBRAL_SIMILITUD,
        umbral_rerank=config.UMBRAL_RERANK,
        fragmentos=[{**_fuente(a), "texto": a["texto"]}
                    for a in resultado["fragmentos"]],
    )


# ---------------------------------------------------------------------------
# POST /ask — pregunta completa: retrieval + generación
# ---------------------------------------------------------------------------

@router.post("/ask", response_model=AskResponse, responses=ERRORES)
async def ask(req: PreguntaRequest):
    traza = uuid.uuid4().hex[:8]
    logger.info("PASO 1 CONSULTA | traza=%s | largo=%d", traza, len(req.pregunta))

    # --- Guardrail de lenguaje (no se busca ni se llama al modelo) ---
    if rag_service.contiene_lenguaje_inapropiado(req.pregunta):
        logger.warning("PASO 2 GUARDRAIL | traza=%s | consulta bloqueada", traza)
        return _respuesta_simple(req.pregunta, CONSULTA_BLOQUEADA)

    # --- Caché: si ya respondimos esta pregunta, devolvemos lo mismo ---
    clave = cache_service.clave(req.pregunta)
    guardada = cache_service.obtener(clave)
    if guardada:
        logger.info("PASO 2 CACHE | traza=%s | respuesta reutilizada", traza)
        guardada = dict(guardada)
        guardada["pregunta"] = req.pregunta
        guardada["desde_cache"] = True
        return AskResponse(**guardada)

    # --- ¿Esta misma pregunta ya pasó por revisión humana? ---
    # Pendiente: no generamos otra. Rechazada: respetamos la decisión.
    # (Las aprobadas ya están en el caché.)
    previa = revision_service.buscar_por_clave(clave)
    if previa and previa["estado"] == revision_service.PENDIENTE:
        logger.info("PASO 2 HITL | traza=%s | sigue pendiente id=%s", traza, previa["id"])
        return _respuesta_en_revision(req.pregunta, previa)
    if previa and previa["estado"] == revision_service.RECHAZADA:
        logger.info("PASO 2 HITL | traza=%s | rechazada antes id=%s", traza, previa["id"])
        return _respuesta_rechazada(req.pregunta, previa)

    # --- Retrieval: búsqueda + filtro de similitud + reranking ---
    try:
        resultado = rag_service.recuperar(req.pregunta, traza=traza)
    except Exception as e:
        logger.error("PASO 3 BUSQUEDA | traza=%s | fallo: %s", traza, type(e).__name__)
        raise HTTPException(status_code=503, detail=SERVICIO_CAIDO)

    if not resultado["pertinente"]:
        return _respuesta_simple(req.pregunta, FUERA_DE_TEMA,
                                 resultado["similarity_score"])

    articulos = resultado["fragmentos"]

    # --- Generación ---
    contexto = rag_service.armar_contexto(articulos)
    try:
        texto = rag_service.generar_respuesta(req.pregunta, contexto)
    except Exception as e:
        logger.error("PASO 5 GENERACION | traza=%s | fallo: %s", traza, type(e).__name__)
        raise HTTPException(status_code=503, detail=SERVICIO_CAIDO)

    texto = rag_service.quitar_emojis(texto)

    # El modelo puede decidir por su cuenta que el contexto no alcanza.
    # Si lo dijo, grounded va en False aunque el score haya pasado.
    fundamentada = rag_service.SIN_CONTEXTO.lower()[:40] not in texto.lower()

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
    motivos = revision_service.evaluar_riesgo(respuesta.model_dump()["fuentes"]) if fundamentada else []
    if motivos:
        id_revision = revision_service.crear(clave, req.pregunta, respuesta.model_dump(), motivos)
        logger.warning(
            "PASO 6 HITL | traza=%s | pending_approval id=%s | motivos=%s",
            traza, id_revision, motivos,
        )
        return _respuesta_en_revision(req.pregunta, revision_service.obtener(id_revision))

    # Guardamos para que la próxima vez la respuesta sea idéntica
    cache_service.guardar(clave, respuesta.model_dump())

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
@router.get("/revisiones/pendientes", response_model=List[RevisionPendiente])
async def revisiones_pendientes():
    """Para el revisor: las respuestas que esperan aprobación, con la propuesta de la IA."""
    return [RevisionPendiente(**r) for r in revision_service.pendientes()]


@router.get("/revisiones/{id_revision}", response_model=EstadoRevision, responses=NO_EXISTE)
async def estado_revision(id_revision: str):
    """Para el usuario: cómo terminó su consulta. La respuesta aparece solo si fue aprobada."""
    return _estado_para_usuario(_revision_o_404(id_revision))


YA_RESUELTA = {409: {"model": ErrorResponse, "description": "Ya fue resuelta"}}


@router.post("/revisiones/{id_revision}/aprobar", response_model=EstadoRevision,
          responses={**NO_EXISTE, **YA_RESUELTA})
async def aprobar(id_revision: str, req: ResolucionRequest):
    """El revisor aprueba: la respuesta se entrega y queda en el caché."""
    revision = _resolver(id_revision, True, req)

    # Desde ahora, la misma pregunta se responde directo con lo aprobado
    cache_service.guardar(revision["clave"], revision["propuesta"])

    return _estado_para_usuario(revision)


@router.post("/revisiones/{id_revision}/rechazar", response_model=EstadoRevision,
          responses={**NO_EXISTE, **YA_RESUELTA})
async def rechazar(id_revision: str, req: ResolucionRequest):
    """El revisor rechaza: la respuesta generada no se entrega nunca."""
    return _estado_para_usuario(_resolver(id_revision, False, req))


def _revision_o_404(id_revision):
    revision = revision_service.obtener(id_revision)
    if not revision:
        raise HTTPException(status_code=404, detail="No existe esa revisión")
    return revision


def _resolver(id_revision, aprobada, req):
    revision = _revision_o_404(id_revision)
    if revision["estado"] != revision_service.PENDIENTE:
        raise HTTPException(status_code=409, detail=f"La revisión ya fue {revision['estado']}")

    revision = revision_service.resolver(id_revision, aprobada, req.revisor, req.comentario)
    logger.info("HITL | revision=%s | %s por revisor", id_revision, revision["estado"])
    return revision


def _estado_para_usuario(revision):
    return EstadoRevision(
        **{k: revision[k] for k in ("id", "estado", "pregunta", "motivos",
                                    "revisor", "comentario", "creada", "resuelta")},
        respuesta=revision["propuesta"] if revision["estado"] == revision_service.APROBADA else None,
    )


# ---------------------------------------------------------------------------
# Raíz
# ---------------------------------------------------------------------------

@router.get("/")
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

"""
schemas.py — Forma de los datos que entran y salen de la API.

FastAPI valida automáticamente con estas clases: una consulta vacía
ni siquiera llega al endpoint.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class PreguntaRequest(BaseModel):
    """POST /ask"""
    pregunta: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Consulta en lenguaje natural sobre la Ley de Contrato de Trabajo",
        examples=["¿Cuántos días de vacaciones me corresponden con 8 años de antigüedad?"],
    )

    @field_validator("pregunta")
    @classmethod
    def sin_espacios_vacios(cls, valor):
        # min_length=1 no alcanza: "   " tiene largo 3 pero está vacío.
        limpio = valor.strip()
        if not limpio:
            raise ValueError("La pregunta no puede estar vacía")
        return limpio


class RetrieveRequest(PreguntaRequest):
    """POST /retrieve"""
    top_k: int = Field(
        3, ge=1, le=10,
        description="Cuántos artículos devolver después del reranking",
    )
    excluir_derogados: bool = Field(
        False,
        description="Si es true, los artículos derogados se filtran en la búsqueda por metadata",
    )


class Fuente(BaseModel):
    """
    Un artículo que el sistema usó para responder.

    Hay dos scores y NO son comparables entre sí:
      - score: similitud de embeddings (qué tan parecido es el texto)
      - score_rerank: relevancia según el reranker (si responde la consulta)
    """
    articulo: str
    titulo: str
    capitulo: str
    derogado: bool
    score: float = Field(..., ge=0.0, le=1.0)
    score_rerank: Optional[float] = Field(None, ge=0.0, le=1.0)


class Fragmento(Fuente):
    """Un fragmento recuperado, con su texto completo (para /retrieve)."""
    texto: str


class RetrieveResponse(BaseModel):
    """
    Resultado del retrieval, sin generación.

    Si pertinente es false, los fragmentos son los mejores candidatos
    igual, para poder ver por qué no alcanzaron los umbrales.
    """
    pregunta: str
    pertinente: bool = Field(
        ..., description="Si hay evidencia suficiente para responder"
    )
    motivo: str = Field(
        ..., description="ok | sin_resultados | similitud_bajo_umbral | rerank_bajo_umbral"
    )
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    umbral_similitud: float
    umbral_rerank: float
    fragmentos: List[Fragmento] = []


class AskResponse(BaseModel):
    """
    Respuesta del asistente.

    Los campos de transparencia (fuentes, similarity_score, grounded,
    aviso) van SIEMPRE, haya respuesta o no.
    """
    pregunta: str
    respuesta: str
    fuentes: List[Fuente] = []
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    grounded: bool = Field(
        ..., description="Indica si la respuesta se apoyó en artículos reales"
    )
    desde_cache: bool = Field(
        False, description="Si la respuesta se reutilizó de una consulta anterior"
    )
    aviso: str


class ErrorResponse(BaseModel):
    error: str

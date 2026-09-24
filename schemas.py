"""
schemas.py — Forma de los datos que entran y salen de la API.

FastAPI valida automáticamente con estas clases: una consulta vacía
ni siquiera llega al endpoint.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ConsultaRequest(BaseModel):
    """POST /consultar"""
    pregunta: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Consulta en lenguaje natural sobre la Ley de Contrato de Trabajo",
    )

    @field_validator("pregunta")
    @classmethod
    def sin_espacios_vacios(cls, valor):
        # min_length=1 no alcanza: "   " tiene largo 3 pero está vacío.
        limpio = valor.strip()
        if not limpio:
            raise ValueError("La pregunta no puede estar vacía")
        return limpio


class ArticuloCitado(BaseModel):
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


class ConsultaResponse(BaseModel):
    """
    Respuesta del asistente.

    Los campos de transparencia (articulos, similarity_score, grounded,
    aviso) van SIEMPRE, haya respuesta o no.
    """
    pregunta: str
    respuesta: str
    articulos: List[ArticuloCitado] = []
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
"""
revisiones.py — ✅📃 Datos que entran y salen de los endpoints de revisión
(Human in the Loop). ✅📃
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from app.schemas.rag import AskResponse


class ResolucionRequest(BaseModel):
    """POST /revisiones/{id}/aprobar y /rechazar"""
    revisor: str = Field(
        ..., min_length=1, max_length=100,
        description="Quién toma la decisión (queda registrado)",
        examples=["Dra. Pérez"],
    )
    comentario: Optional[str] = Field(
        None, max_length=500,
        examples=["El art. 173 está derogado: no aplica."],
    )


class RevisionPendiente(BaseModel):
    """Lo que ve la Persona / El Humano: incluye la respuesta propuesta por la IA.👨‍💻"""
    id: str
    pregunta: str
    motivos: List[str]
    propuesta: AskResponse
    creada: str


class EstadoRevision(BaseModel):
    """
    Lo que ve el usuario. 🌐 La respuesta solo aparece si fue aprobada:
    mientras está pendiente, el texto generado no sale de la API. 🔑
    """
    id: str
    estado: Literal["pendiente", "aprobada", "rechazada"]
    pregunta: str
    motivos: List[str]
    respuesta: Optional[AskResponse] = None
    revisor: Optional[str] = None
    comentario: Optional[str] = None
    creada: str
    resuelta: Optional[str] = None

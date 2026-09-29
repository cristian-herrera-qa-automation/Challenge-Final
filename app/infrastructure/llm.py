"""
llm.py — 🤖🔑 Conexión con Cohere. 🤖🔑

Es el único archivo que habla con la API de Cohere. El resto del
proyecto pide embeddings, reranking o una respuesta del modelo acá.
"""

import os

import cohere

from app.config import MODELO_EMBEDDINGS, MODELO_CHAT, MODELO_RERANK


def _crear_cliente_cohere():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("Falta COHERE_API_KEY en el archivo .env")
    return cohere.ClientV2(api_key=api_key)


co = _crear_cliente_cohere()


def embeber(textos, tipo):
    """
    Convierte textos en vectores.

    tipo es "search_document" para la ley (en la ingesta) y
    "search_query" para la consulta del usuario.
    """
    respuesta = co.embed(
        texts=textos,
        model=MODELO_EMBEDDINGS,
        input_type=tipo,
        embedding_types=["float"],
    )
    return respuesta.embeddings.float


def rerank(consulta, documentos, top_n):
    """Devuelve [(índice del documento, score)], de más a menos relevante."""
    respuesta = co.rerank(
        model=MODELO_RERANK,
        query=consulta,
        documents=documentos,
        top_n=top_n,
    )
    return [(item.index, item.relevance_score) for item in respuesta.results]


def chat(mensajes, temperatura, **opciones):
    """Le manda los mensajes al modelo y devuelve el texto de la respuesta."""
    respuesta = co.chat(
        model=MODELO_CHAT,
        messages=mensajes,
        temperature=temperatura,
        **opciones,
    )
    return respuesta.message.content[0].text

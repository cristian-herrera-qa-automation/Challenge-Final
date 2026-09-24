"""
rag.py — Lógica del asistente sobre la Ley de Contrato de Trabajo.

Este archivo NO carga documentos: eso lo hizo ingesta.py una sola vez.
Acá solo se consulta la base vectorial que ya está en disco.
"""

import os
import re
import logging

import cohere
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

MODELO_EMBEDDINGS = "embed-multilingual-v3.0"
MODELO_CHAT = "command-a-03-2025"
MODELO_RERANK = "rerank-v4.0-fast"

# Cuántos artículos se recuperan por similitud antes de reordenar.
CANDIDATOS = 10

# Cuántos quedan finalmente como contexto para el modelo.
TOP_K = 3

# Umbral de pertinencia sobre la similitud de embeddings.
# Filtro barato: decide si la consulta es sobre la ley, antes de
# gastar una llamada al reranker.
# Medido: consultas pertinentes 0.56 a 0.72, ajenas 0.33 a 0.34.
UMBRAL_SIMILITUD = 0.45

# Umbral sobre el score del reranker. Es OTRA escala, no comparable
# con la de arriba.
# Medido: artículos pertinentes 0.57 a 0.91, consultas ajenas 0.09 a 0.11.
UMBRAL_RERANK = 0.30

# temperature=0 es lo que hace la respuesta reproducible.
TEMPERATURA = 0.0

CARPETA_CHROMA = "./chroma_data"
COLECCION = "ley_laboral"


def _crear_cliente_cohere():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("Falta COHERE_API_KEY en el archivo .env")
    return cohere.ClientV2(api_key=api_key)


co = _crear_cliente_cohere()

chroma_client = chromadb.PersistentClient(
    path=CARPETA_CHROMA,
    settings=Settings(anonymized_telemetry=False),
)

try:
    coleccion = chroma_client.get_collection(name=COLECCION)
except Exception:
    raise RuntimeError(
        f"No existe la colección '{COLECCION}' en {CARPETA_CHROMA}.\n"
        "Corré primero:  python ingesta.py"
    )


# ---------------------------------------------------------------------------
# 1. Guardrail de lenguaje
# ---------------------------------------------------------------------------

PALABRAS_BLOQUEADAS = {
    "idiota", "estupido", "estúpido", "imbecil", "imbécil",
    "odio", "inferior", "inferiores", "insulto", "ofensa",
}


def contiene_lenguaje_inapropiado(texto):
    """
    True si la consulta trae lenguaje bloqueado.

    Usa \\b (límite de palabra) para no marcar "custodio" por contener
    "odio", ni "inferior" dentro de "inferioridad" en un uso legítimo.
    """
    minuscula = texto.lower()
    for palabra in PALABRAS_BLOQUEADAS:
        if re.search(rf"\b{re.escape(palabra)}\b", minuscula):
            return True
    return False


# ---------------------------------------------------------------------------
# 2. Búsqueda en la base vectorial
# ---------------------------------------------------------------------------

def _distancia_a_similitud(distancia):
    """Chroma devuelve distancia coseno (0 a 2). La pasamos a 0-1."""
    return max(0.0, min(1.0, 1.0 - distancia))


def buscar(consulta, top_k=CANDIDATOS):
    """
    Busca los artículos más relevantes para la consulta.

    Devuelve una lista de diccionarios con el texto del artículo,
    su metadata y el score de similitud.
    """
    respuesta = co.embed(
        texts=[consulta],
        model=MODELO_EMBEDDINGS,
        input_type="search_query",
        embedding_types=["float"],
    )
    vector = respuesta.embeddings.float[0]

    # Pedimos de más porque después descartamos fragmentos repetidos
    # del mismo artículo.
    resultados = coleccion.query(
        query_embeddings=[vector],
        n_results=top_k * 3,
    )

    if not resultados["ids"] or not resultados["ids"][0]:
        return []

    vistos = {}

    for i in range(len(resultados["ids"][0])):
        meta = resultados["metadatas"][0][i]
        clave = meta["articulo"]

        # Chroma devuelve ordenado de mejor a peor: nos quedamos
        # con el primer fragmento de cada artículo.
        if clave in vistos:
            continue

        vistos[clave] = {
            "articulo": meta["articulo"],
            "titulo": meta["titulo"],
            "titulo_nombre": meta["titulo_nombre"],
            "capitulo": meta["capitulo"],
            "capitulo_nombre": meta["capitulo_nombre"],
            "derogado": meta["derogado"],
            "texto": resultados["documents"][0][i],
            "score": round(_distancia_a_similitud(resultados["distances"][0][i]), 4),
        }

    return list(vistos.values())[:top_k]


# ---------------------------------------------------------------------------
# 3. Reranking
# ---------------------------------------------------------------------------

def reordenar(consulta, articulos, top_n=TOP_K):
    """
    Reordena los artículos por relevancia real usando el reranker.

    Diferencia con la búsqueda: el embedding compara la consulta contra
    un resumen numérico del artículo, calculado por separado y de
    antemano. El reranker lee consulta y artículo JUNTOS, así que
    distingue "período de prueba" de "prueba del contrato" aunque
    compartan la palabra.

    Si el reranker falla, se devuelve el orden original. Una mejora
    de calidad no debe tumbar todo el servicio.
    """
    if not articulos:
        return []

    try:
        respuesta = co.rerank(
            model=MODELO_RERANK,
            query=consulta,
            documents=[a["texto"] for a in articulos],
            top_n=min(top_n, len(articulos)),
        )
    except Exception as e:
        logger.warning("Rerank no disponible (%s). Se usa el orden por similitud.",
                       type(e).__name__)
        return articulos[:top_n]

    reordenados = []
    for item in respuesta.results:
        art = dict(articulos[item.index])
        art["score_rerank"] = round(item.relevance_score, 4)
        reordenados.append(art)

    return reordenados


# ---------------------------------------------------------------------------
# 4. Armado del contexto
# ---------------------------------------------------------------------------

def armar_contexto(articulos):
    """
    Junta los artículos recuperados en un solo texto para el modelo.

    Cada bloque lleva su número de artículo y, si corresponde, el aviso
    de que está derogado. Así el modelo puede citarlo y advertirlo.
    """
    bloques = []

    for art in articulos:
        estado = " [ARTICULO DEROGADO]" if art["derogado"] else ""
        bloques.append(
            f"--- Artículo {art['articulo']} de la Ley 20.744{estado} ---\n"
            f"{art['texto']}"
        )

    return "\n\n".join(bloques)


# ---------------------------------------------------------------------------
# 5. Generación de la respuesta
# ---------------------------------------------------------------------------

SIN_CONTEXTO = (
    "No cuento con información suficiente en la Ley de Contrato de Trabajo "
    "para responder a esta consulta."
)

INSTRUCCIONES = f"""Sos un asistente que responde consultas sobre la Ley de Contrato de Trabajo argentina (Ley 20.744).

REGLAS QUE DEBES CUMPLIR SIEMPRE:

1. Respondé ÚNICAMENTE con la información del contexto. No agregues conocimiento propio ni interpretaciones.
2. Si la respuesta no está en el contexto, respondé exactamente: "{SIN_CONTEXTO}"
3. Citá siempre el número de artículo en el que te basás. Ejemplo: "según el artículo 150".
4. Si un artículo del contexto figura como DEROGADO, advertilo explícitamente.
5. Respondé SIEMPRE en español, sin importar el idioma de la pregunta.
6. No uses emojis ni símbolos decorativos.
7. No des consejos legales ni opiniones. Limitate a informar qué dice la ley.
8. No hagas juicios de valor sobre empleadores ni trabajadores.
9. Sé claro y conciso. Si la ley establece plazos o montos, indicalos con precisión."""


def generar_respuesta(pregunta, contexto):
    """Le pide al modelo que responda usando solo el contexto recuperado."""
    prompt = f"""{INSTRUCCIONES}

CONTEXTO (artículos de la Ley 20.744):
{contexto}

CONSULTA: {pregunta}

RESPUESTA:"""

    respuesta = co.chat(
        model=MODELO_CHAT,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURA,
    )

    return respuesta.message.content[0].text.strip()


# ---------------------------------------------------------------------------
# 6. Limpieza de la salida
# ---------------------------------------------------------------------------

# Rangos Unicode de emojis y pictogramas.
PATRON_EMOJIS = re.compile(
    "["
    "\U0001F300-\U0001FAFF"   # símbolos, pictogramas, emoticones
    "\U00002600-\U000027BF"   # símbolos varios y dingbats
    "\U0001F000-\U0001F0FF"   # fichas de juego
    "\U00002B00-\U00002BFF"   # flechas
    "\U0000FE00-\U0000FE0F"   # selectores de variación
    "\U0001F1E6-\U0001F1FF"   # banderas
    "]+",
    flags=re.UNICODE,
)


def quitar_emojis(texto):
    """
    Saca emojis de la respuesta.

    El prompt ya se lo pide al modelo, pero esto lo garantiza:
    una instrucción se puede desobedecer, un filtro no.
    """
    limpio = PATRON_EMOJIS.sub("", texto)
    return re.sub(r"[ ]{2,}", " ", limpio).strip()
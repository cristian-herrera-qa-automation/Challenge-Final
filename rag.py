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
# Se pide de más a propósito: el embedding no siempre deja el artículo
# correcto arriba (caso real: en "período de prueba" entraba el art. 50,
# que trata la prueba del contrato). Con 10 candidatos el reranker
# tiene margen para reordenar.
CANDIDATOS = 10

# Cuántos quedan finalmente como contexto para el modelo.
# Las consultas típicas se responden con 1 artículo, o con 2 que se
# remiten entre sí (ej. 92 bis y 231). Más contexto aumenta el riesgo
# de que el modelo cite artículos que no responden la consulta.
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


def buscar(consulta, top_k=CANDIDATOS, excluir_derogados=False):
    """
    Busca los artículos más relevantes para la consulta.

    Devuelve una lista de diccionarios con el texto del artículo,
    su metadata y el score de similitud.

    Con excluir_derogados=True se usa la metadata para filtrar en la
    propia búsqueda: los artículos derogados ni siquiera compiten.
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
        where={"derogado": False} if excluir_derogados else None,
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
# 3 bis. Retrieval completo (búsqueda + filtros + reranking)
# ---------------------------------------------------------------------------

def recuperar(consulta, top_k=TOP_K, excluir_derogados=False, traza="-"):
    """
    Ejecuta todo el retrieval SIN llamar al modelo generativo.

    Lo usan /retrieve (para inspeccionar qué se recupera) y /ask (para
    responder), así los dos ven exactamente los mismos fragmentos.

    Devuelve un diccionario:
      - pertinente:  si hay evidencia suficiente para responder
      - motivo:      por qué se decidió eso
      - fragmentos:  si es pertinente, los que pasaron los dos filtros;
                     si no, los mejores candidatos, para poder ver por qué
      - similarity_score: el mejor score de similitud
    """
    candidatos = buscar(consulta, excluir_derogados=excluir_derogados)

    if not candidatos:
        logger.info("PASO 3 BUSQUEDA | traza=%s | sin resultados", traza)
        return _resultado(False, "sin_resultados", [], 0.0)

    mejor = candidatos[0]
    logger.info(
        "PASO 3 BUSQUEDA | traza=%s | %d candidatos | mejor=%s score=%.4f",
        traza, len(candidatos), mejor["articulo"], mejor["score"],
    )

    # Filtro 1 (barato): ¿la consulta es sobre la ley? Si no supera el
    # umbral, cortamos acá y nos ahorramos el reranker y el modelo.
    if mejor["score"] < UMBRAL_SIMILITUD:
        logger.info(
            "PASO 4 PERTINENCIA | traza=%s | score=%.4f < umbral=%.2f | fuera de tema",
            traza, mejor["score"], UMBRAL_SIMILITUD,
        )
        return _resultado(False, "similitud_bajo_umbral",
                          candidatos[:top_k], mejor["score"])

    # Filtro 2: de los candidatos, cuáles responden REALMENTE
    reordenados = reordenar(consulta, candidatos, top_n=top_k)
    relevantes = [a for a in reordenados
                  if a.get("score_rerank", 1.0) >= UMBRAL_RERANK]

    if not relevantes:
        logger.info("PASO 4 RERANK | traza=%s | ningun articulo relevante", traza)
        return _resultado(False, "rerank_bajo_umbral", reordenados, mejor["score"])

    logger.info(
        "PASO 4 RERANK | traza=%s | %s",
        traza, [(a["articulo"], a.get("score_rerank")) for a in relevantes],
    )
    return _resultado(True, "ok", relevantes, mejor["score"])


def _resultado(pertinente, motivo, fragmentos, score):
    return {
        "pertinente": pertinente,
        "motivo": motivo,
        "fragmentos": fragmentos,
        "similarity_score": score,
    }


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

# System prompt: define rol, uso de evidencia, restricciones, formato y
# qué hacer cuando no hay información. Va separado de la consulta del
# usuario (role "system"), así lo que escriba el usuario no se mezcla
# con las reglas.
INSTRUCCIONES = f"""ROL
Sos un asistente que informa qué dice la Ley de Contrato de Trabajo argentina (Ley 20.744). Tus usuarios son trabajadores y empleadores sin formación jurídica.

USO DE LA EVIDENCIA
1. Respondé ÚNICAMENTE con la información de los artículos del CONTEXTO. No agregues conocimiento propio, otras leyes, convenios colectivos ni interpretaciones.
2. Citá siempre el número de artículo en el que te basás. Ejemplo: "según el artículo 150".
3. Si un artículo del contexto figura como DEROGADO, advertilo explícitamente y no lo presentes como vigente.

CUANDO NO HAY INFORMACIÓN SUFICIENTE
4. Si el contexto no alcanza para responder, respondé exactamente: "{SIN_CONTEXTO}" No completes con suposiciones.

RESTRICCIONES
5. Respondé SIEMPRE en español, sin importar el idioma de la pregunta.
6. No des consejos legales ni opiniones. Limitate a informar qué dice la ley.
7. No hagas juicios de valor sobre empleadores ni trabajadores.
8. No uses emojis ni símbolos decorativos.

FORMATO DE LA RESPUESTA
9. Primera oración: la respuesta directa a la consulta, con el artículo citado.
10. Después, si hace falta, los detalles (plazos, montos, condiciones) en uno o dos párrafos breves. Si la ley establece plazos o montos, indicalos con precisión.
11. Texto plano, sin títulos ni tablas. Como máximo 150 palabras."""


def generar_respuesta(pregunta, contexto):
    """Le pide al modelo que responda usando solo el contexto recuperado."""
    mensaje_usuario = f"""CONTEXTO (artículos de la Ley 20.744):
{contexto}

CONSULTA: {pregunta}"""

    # Si el modelo devuelve una salida degenerada, se reintenta. El primer
    # intento va con temperature=0; los reintentos con una temperatura
    # baja, porque en la evaluación la misma pregunta degeneró dos veces
    # seguidas con 0: la idea es sacar al modelo del bucle. El determinismo
    # no se pierde: la respuesta que se entrega queda fija en el caché.
    # Si falla las tres veces, se levanta un error: es preferible un 503
    # a entregar (y guardar en el caché) una respuesta basura.
    for intento, temperatura in enumerate(TEMPERATURAS_POR_INTENTO, start=1):
        respuesta = co.chat(
            model=MODELO_CHAT,
            messages=[
                {"role": "system", "content": INSTRUCCIONES},
                {"role": "user", "content": mensaje_usuario},
            ],
            temperature=temperatura,
            max_tokens=MAX_TOKENS,
        )
        texto = respuesta.message.content[0].text.strip()

        if not es_degenerada(texto):
            return texto
        logger.warning("Salida degenerada del modelo (intento %d, temp=%.1f, %d caracteres, empieza %r)",
                       intento, temperatura, len(texto), texto[:30])

    raise RuntimeError("El modelo devolvió una salida degenerada en todos los intentos")


TEMPERATURAS_POR_INTENTO = (TEMPERATURA, 0.3, 0.3)


# Caso real observado: con temperature=0, command-a devolvió una vez
# "Según el artículo 201..." reemplazado por miles de "3" seguidos.
# No se pudo reproducir, así que no se puede evitar: hay que detectarlo.
# Arranca en un carácter que no sea espacio, para no marcar sangrías.
PATRON_REPETICION = re.compile(r"(\S.{0,9}?)\1{15,}", re.DOTALL)

# 150 palabras en español son unos 250 tokens: 400 deja margen para una
# respuesta normal y corta un bucle antes de que crezca.
MAX_TOKENS = 400


def es_degenerada(texto):
    """True si la salida es un bucle de repetición (ej. "3333..." o "abcabc...")."""
    return not texto or bool(PATRON_REPETICION.search(texto))


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
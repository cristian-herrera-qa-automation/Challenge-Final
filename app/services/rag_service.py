"""
rag_service.py — Lógica del asistente sobre la Ley de Contrato de Trabajo.

Guardrail, retrieval (búsqueda + filtros + reranking), armado del
contexto, prompt y generación de la respuesta.

"""

import re
import logging

from app.config import (
    CANDIDATOS, TOP_K, UMBRAL_SIMILITUD, UMBRAL_RERANK, TEMPERATURA,
)
from app.infrastructure import llm, vector_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 🚨🚨1. Guardrail de lenguaje 🚨🚨
# ---------------------------------------------------------------------------

PALABRAS_BLOQUEADAS = {
    "idiota", "estupido", "estúpido", "imbecil", "imbécil",
    "odio", "inferior", "inferiores", "insulto", "ofensa",
}


def contiene_lenguaje_inapropiado(texto):
    """
    True si la consulta trae lenguaje bloqueado. 🚨✋

    Usa \\b (límite de palabra) para no marcar "custodio" por contener
    "odio", ni "inferior" dentro de "inferioridad" en un uso legítimo.
    """
    minuscula = texto.lower()
    for palabra in PALABRAS_BLOQUEADAS:
        if re.search(rf"\b{re.escape(palabra)}\b", minuscula):
            return True
    return False


# ---------------------------------------------------------------------------
# 🔍🔍2. Búsqueda en la base vectorial 🔍🔍
# ---------------------------------------------------------------------------

def buscar(consulta, top_k=CANDIDATOS, excluir_derogados=False):
    """
    🧪✅🔍Busca los articulos más relevantes para la consulta.

    Convierte la consulta en vector y le pide a la base vectorial los
    artículos más parecidos, con su metadata y el score de similitud. 🧪✅🔍
    """
    vector = llm.embeber([consulta], tipo="search_query")[0]
    return vector_store.buscar_por_vector(vector, top_k, excluir_derogados)


# ---------------------------------------------------------------------------
# 🔟🔍📃 3. Reranking 🔍📃🔟
# ---------------------------------------------------------------------------

def reordenar(consulta, articulos, top_n=TOP_K):
    """
    Reordena los artículos por relevancia real usando el reranker. 🔟📃

    La diferencia con la búsqueda es q el embedding compara la consulta contra
    un resumen numérico del artículo.
    El reranker lee consulta y artículo JUNTOS, asi que
    distingue por ejemplo "período de prueba" de "prueba del contrato" aunque
    compartan la palabra. 📃✅

    Si una llamada al reranker falla, se devuelve el orden original. La idea es que esto
    no afecte a todo el sistema. ✋
    """
    if not articulos:
        return []

    try:
        resultados = llm.rerank(
            consulta,
            [a["texto"] for a in articulos],
            top_n=min(top_n, len(articulos)),
        )
    except Exception as e:
        logger.warning("Rerank no disponible (%s). Se usa el orden por similitud.",
                       type(e).__name__)
        return articulos[:top_n]

    reordenados = []
    for indice, score in resultados:
        art = dict(articulos[indice])
        art["score_rerank"] = round(score, 4)
        reordenados.append(art)

    return reordenados


# ---------------------------------------------------------------------------
# 🔍📃 3 bis. Retrieval completo (búsqueda + filtros + reranking) 🔍📃
# ---------------------------------------------------------------------------

def recuperar(consulta, top_k=TOP_K, excluir_derogados=False, traza="-", candidatos=None):
    """
    Ejecuta todo el retrieval SIN llamar al LLM. 📃👨‍💻

    Lo usan el Endpoint /retrieve (para ver que es lo que recupera) y /ask (para
    responder), asi los dos ven exactamente los mismos fragmentos. 📃👨‍💻

    Devuelve un diccionario:
      - pertinente:  si hay evidencia suficiente para responder
      - motivo:      porqué se decidió eso
      - fragmentos:  si es pertinente, los que pasaron los dos filtros;
                     si no, los mejores candidatos, para poder ver el porqué
      - similarity_score: el mejor score de similitud

    candidatos permite pasar una búsqueda ya hecha (lo usa la evaluación,
    para no pagar dos veces el mismo embedding).
    """
    if candidatos is None:
        candidatos = buscar(consulta, excluir_derogados=excluir_derogados)

    if not candidatos:
        logger.info("PASO 3 BUSQUEDA | traza=%s | sin resultados", traza)
        return _resultado(False, "sin_resultados", [], 0.0)

    mejor = candidatos[0]
    logger.info(
        "PASO 3 BUSQUEDA | traza=%s | %d candidatos | mejor=%s score=%.4f",
        traza, len(candidatos), mejor["articulo"], mejor["score"],
    )

    # Filtro 1 : ¿la consulta es sobre la ley? Si no supera el
    # umbral, cortamos acá y nos ahorramos el reranker y el modelo. 🔍✅
    if mejor["score"] < UMBRAL_SIMILITUD:
        logger.info(
            "PASO 4 PERTINENCIA | traza=%s | score=%.4f < umbral=%.2f | fuera de tema",
            traza, mejor["score"], UMBRAL_SIMILITUD,
        )
        return _resultado(False, "similitud_bajo_umbral",
                          candidatos[:top_k], mejor["score"])

    # Filtro 2: de los candidatos, cuáles responden REALMENTE 🔟✅
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
# 📃 4. Armado del contexto 📃
# ---------------------------------------------------------------------------

def armar_contexto(articulos):
    """
    Junta los artículos recuperados en un solo texto para el modelo. 📃✅

    Cada bloque lleva su número de artículo y, si corresponde, el aviso
    de que está derogado. Así el modelo puede citarlo y advertirlo. 🤖✋
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
# 🤖🧪 5. Generación de la respuesta 🤖🧪
# ---------------------------------------------------------------------------

SIN_CONTEXTO = (
    "No cuento con información suficiente en la Ley de Contrato de Trabajo "
    "para responder a esta consulta."
)

# System prompt: define rol, uso de evidencia, restricciones, formato y
# qué hacer cuando no hay información. 🤖✋✅

INSTRUCCIONES = f"""ROL
Sos un asistente que informa que es lo que dice la Ley de Contrato de Trabajo argentina (Ley 20.744). Tus usuarios son trabajadores y empleadores sin formación jurídica.

USO DE LA EVIDENCIA
1. Respondé ÚNICAMENTE con la información de los artículos del CONTEXTO. No agregues conocimiento propio, otras leyes, convenios colectivos ni interpretaciones.
2. No agregues explicaciones ni consecuencias que no estén escritas en los artículos del CONTEXTO: ni para qué sirve una norma, ni qué pasa si no se cumple, ni datos de otros artículos que no te pasaron. Aunque sean ciertas, si no están escritas en el contexto, no van.
3. Citá siempre el número de artículo en el que te basás. Ejemplo: "según el artículo 150". Cada dato tiene que salir del artículo que citás.
4. Si un artículo del contexto figura como DEROGADO, advertilo explícitamente y no lo presentes como vigente.

CUANDO NO HAY INFORMACIÓN SUFICIENTE
5. Si el contexto no alcanza para responder, respondé exactamente: "{SIN_CONTEXTO}" No completes con suposiciones.

RESTRICCIONES
6. Respondé SIEMPRE en español, sin importar el idioma de la pregunta.
7. No des consejos legales ni opiniones. Limitate a informar lo que dice la ley.
8. No hagas juicios de valor sobre empleadores ni trabajadores.
9. No uses emojis ni símbolos decorativos.

FORMATO DE LA RESPUESTA
10. Primera oración: la respuesta directa a la consulta, con el artículo citado.
11. Después, solo si el artículo los trae, los detalles (plazos, montos, condiciones) en uno o dos párrafos breves. Si la ley establece plazos o montos, indicalos con precisión.
12. Texto plano, sin títulos ni tablas. Como máximo 150 palabras."""


def generar_respuesta(pregunta, contexto):
    """📃🚀 Le pide al modelo que responda usando solo el contexto recuperado. 📃🚀 """ 
    mensaje_usuario = f"""CONTEXTO (artículos de la Ley 20.744):
{contexto}

CONSULTA: {pregunta}"""


    for intento, temperatura in enumerate(TEMPERATURAS_POR_INTENTO, start=1):
        texto = llm.chat(
            [
                {"role": "system", "content": INSTRUCCIONES},
                {"role": "user", "content": mensaje_usuario},
            ],
            temperatura,
            max_tokens=MAX_TOKENS,
        ).strip()

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
# respuesta normal y corta un bucle antes de que crezca. 🤖✋
MAX_TOKENS = 400


def es_degenerada(texto):
    """True si la salida es un bucle de repetición (ej. "3333..." o "abcabc...")."""
    return not texto or bool(PATRON_REPETICION.search(texto))


# ---------------------------------------------------------------------------
# 🤖🧪 6. Limpieza de la salida 🤖🧪
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
    Saca emojis de la respuesta. 🚨✋

    El prompt ya se lo pide al modelo, pero esto algo que lo garantiza:
    una instrucción quizas no es tan fuerte, un filtro no. 🚨✋
    """
    limpio = PATRON_EMOJIS.sub("", texto)
    return re.sub(r"[ ]{2,}", " ", limpio).strip()
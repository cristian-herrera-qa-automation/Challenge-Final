"""
evaluar.py — Evaluación del RAG sobre el dataset de 15 preguntas.

    python eval/evaluar.py

Evalúa tres cosas por separado:

  1. Retrieval: ¿se recuperó el artículo que responde la pregunta?
     Se compara solo embeddings contra embeddings + reranking.
  2. Comportamiento: ¿el sistema hizo lo que tenía que hacer? (responder,
     negarse, cortar por fuera de tema o mandar a revisión humana)
  3. Generación: ¿la respuesta es correcta, relevante y está respaldada
     por el contexto? Con un chequeo automático de datos clave y un
     LLM-as-a-Judge.

No pasa por la API ni por el caché: llama directo a las funciones de
rag.py, así cada corrida mide al sistema de verdad y no deja revisiones
ni respuestas guardadas.

Guarda todo en eval/resultados.json.
"""

import os
import re
import sys
import json
import time
import logging
import unicodedata
from datetime import datetime

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RAIZ)  # rag.py abre ./chroma_data con ruta relativa
sys.path.insert(0, RAIZ)

from dotenv import load_dotenv

load_dotenv(os.path.join(RAIZ, ".env"))

import rag
import revisiones

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

DATASET = "eval/dataset.json"
SALIDA = "eval/resultados.json"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def normalizar(texto):
    """Minúsculas y sin tildes, para comparar datos clave."""
    texto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def metricas_retrieval(recuperados, esperados):
    """
    hit:    ¿apareció al menos un artículo esperado en el top 3?
    recall: qué fracción de los esperados apareció
    rr:     1 / posición del primer esperado (1 si está primero, 0 si no está)
    """
    hit = any(a in recuperados for a in esperados)
    recall = sum(a in recuperados for a in esperados) / len(esperados)
    rr = 0.0
    for posicion, articulo in enumerate(recuperados, start=1):
        if articulo in esperados:
            rr = 1 / posicion
            break
    return {"hit": hit, "recall": round(recall, 3), "rr": round(rr, 3)}


def cita_articulo(texto, articulo):
    """¿La respuesta menciona el número de artículo? ("150", "92 bis")."""
    return re.search(rf"\b{re.escape(articulo)}\b", normalizar(texto)) is not None


# ---------------------------------------------------------------------------
# LLM-as-a-Judge
# ---------------------------------------------------------------------------

JUEZ = """Sos un evaluador estricto de un asistente que responde sobre la Ley de Contrato de Trabajo argentina (Ley 20.744).

Vas a recibir la PREGUNTA, el CONTEXTO (los artículos que el sistema recuperó), la RESPUESTA del asistente y los DATOS ESPERADOS (lo que una respuesta correcta debería incluir; puede estar vacío).

Puntuá de 1 a 5 cada criterio:
- correcta: la información es correcta según el contexto y coincide con los datos esperados. 5 = totalmente correcta. 1 = incorrecta.
- relevante: responde exactamente lo que se preguntó, sin irse por las ramas. 5 = responde justo eso. 1 = no responde la pregunta.
- fundamentada: todo lo que afirma está escrito en el contexto. 5 = nada inventado. 1 = afirma cosas que el contexto no dice, o saca conclusiones que el artículo citado no respalda.

Devolvé SOLO un JSON con esta forma:
{"correcta": n, "relevante": n, "fundamentada": n, "comentario": "una oración explicando el puntaje más bajo"}"""


def juzgar(pregunta, contexto, respuesta, datos_clave):
    esperados = "; ".join(" o ".join(alternativas) for alternativas in datos_clave) or "(sin datos esperados)"
    mensaje = (
        f"PREGUNTA: {pregunta}\n\n"
        f"CONTEXTO:\n{contexto}\n\n"
        f"RESPUESTA: {respuesta}\n\n"
        f"DATOS ESPERADOS: {esperados}"
    )
    salida = rag.co.chat(
        model=rag.MODELO_CHAT,
        messages=[
            {"role": "system", "content": JUEZ},
            {"role": "user", "content": mensaje},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(salida.message.content[0].text)


# ---------------------------------------------------------------------------
# Evaluación de una pregunta
# ---------------------------------------------------------------------------

def evaluar(caso):
    pregunta = caso["pregunta"]
    esperados = caso["articulos_esperados"]
    r = {"id": caso["id"], "categoria": caso["categoria"], "pregunta": pregunta,
         "articulos_esperados": esperados,
         "comportamiento_esperado": caso["comportamiento_esperado"]}

    # --- 1. Retrieval: solo embeddings vs. embeddings + reranking ---
    # Una sola búsqueda: se usa para las dos variantes (ahorra llamadas)
    candidatos = rag.buscar(pregunta)
    solo_embeddings = [a["articulo"] for a in candidatos[:rag.TOP_K]]
    resultado = rag.recuperar(pregunta, candidatos=candidatos)
    finales = [a["articulo"] for a in resultado["fragmentos"]]

    r["retrieval"] = {
        "solo_embeddings": solo_embeddings,
        "con_rerank": finales if resultado["pertinente"] else [],
        "similarity_score": resultado["similarity_score"],
        "scores_rerank": [a.get("score_rerank") for a in resultado["fragmentos"]],
        "derogados": [a["articulo"] for a in resultado["fragmentos"] if a["derogado"]],
        "motivo": resultado["motivo"],
    }
    if esperados:
        r["retrieval"]["metricas_solo_embeddings"] = metricas_retrieval(solo_embeddings, esperados)
        r["retrieval"]["metricas_con_rerank"] = metricas_retrieval(r["retrieval"]["con_rerank"], esperados)

    # --- 2. Comportamiento (misma lógica que /ask, sin caché) ---
    r["respuesta"] = None
    if not resultado["pertinente"]:
        r["comportamiento"] = "fuera_de_tema"
    else:
        contexto = rag.armar_contexto(resultado["fragmentos"])
        try:
            texto = rag.quitar_emojis(rag.generar_respuesta(pregunta, contexto))
        except Exception as e:
            if type(e).__name__ == "TooManyRequestsError":
                raise  # sin cupo en Cohere: mejor cortar que guardar resultados vacíos
            r["comportamiento"] = "error"
            r["error"] = type(e).__name__
            texto = None

        if texto is not None:
            r["respuesta"] = texto
            fundamentada = rag.SIN_CONTEXTO.lower()[:40] not in texto.lower()
            fuentes = [{"derogado": a["derogado"], "score_rerank": a.get("score_rerank")}
                       for a in resultado["fragmentos"]]
            motivos = revisiones.evaluar_riesgo(fuentes) if fundamentada else []
            r["motivos_revision"] = motivos

            if not fundamentada:
                r["comportamiento"] = "no_responder"
            elif motivos:
                r["comportamiento"] = "revision"
            else:
                r["comportamiento"] = "responder"

            # --- 3. Generación: solo si el modelo produjo una respuesta ---
            if fundamentada:
                r["datos_clave"] = [
                    any(normalizar(alt) in normalizar(texto) for alt in alternativas)
                    for alternativas in caso["datos_clave"]
                ]
                r["cita_articulo_esperado"] = (
                    any(cita_articulo(texto, a) for a in esperados) if esperados else None
                )
                try:
                    r["juez"] = juzgar(pregunta, contexto, texto, caso["datos_clave"])
                except Exception as e:
                    r["juez"] = {"error": type(e).__name__}

    r["comportamiento_ok"] = r["comportamiento"] == caso["comportamiento_esperado"]
    return r


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------

def promedio(valores):
    valores = list(valores)
    return round(sum(valores) / len(valores), 3) if valores else None


def resumir(resultados):
    con_esperados = [r for r in resultados if "metricas_con_rerank" in r["retrieval"]]
    resumen = {"preguntas": len(resultados), "retrieval": {}}

    for modo in ("solo_embeddings", "con_rerank"):
        m = [r["retrieval"][f"metricas_{modo}"] for r in con_esperados]
        resumen["retrieval"][modo] = {
            "preguntas_evaluadas": len(m),
            "hit_rate@3": promedio(x["hit"] for x in m),
            "recall@3": promedio(x["recall"] for x in m),
            "mrr": promedio(x["rr"] for x in m),
        }

    resumen["comportamiento_correcto"] = f"{sum(r['comportamiento_ok'] for r in resultados)}/{len(resultados)}"

    juzgadas = [r for r in resultados if "juez" in r and "error" not in r["juez"]]
    resumen["juez"] = {
        "respuestas_juzgadas": len(juzgadas),
        "correcta": promedio(r["juez"]["correcta"] for r in juzgadas),
        "relevante": promedio(r["juez"]["relevante"] for r in juzgadas),
        "fundamentada": promedio(r["juez"]["fundamentada"] for r in juzgadas),
    }

    con_datos = [r for r in resultados if r.get("datos_clave")]
    resumen["datos_clave_presentes"] = promedio(all(r["datos_clave"]) for r in con_datos)
    con_cita = [r for r in resultados if r.get("cita_articulo_esperado") is not None]
    resumen["cita_articulo_esperado"] = promedio(r["cita_articulo_esperado"] for r in con_cita)
    return resumen


# ---------------------------------------------------------------------------

def main():
    casos = json.load(open(DATASET, encoding="utf-8"))
    resultados = []

    for caso in casos:
        print(f"[{caso['id']:2}/{len(casos)}] {caso['pregunta']}", flush=True)
        resultados.append(evaluar(caso))
        time.sleep(7)  # la clave Trial permite 10 reranks y 20 chats por minuto

    salida = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "configuracion": {
            "modelo_embeddings": rag.MODELO_EMBEDDINGS,
            "modelo_chat": rag.MODELO_CHAT,
            "modelo_rerank": rag.MODELO_RERANK,
            "modelo_juez": rag.MODELO_CHAT,
            "top_k": rag.TOP_K,
            "candidatos": rag.CANDIDATOS,
            "umbral_similitud": rag.UMBRAL_SIMILITUD,
            "umbral_rerank": rag.UMBRAL_RERANK,
            "umbral_confianza_hitl": revisiones.UMBRAL_CONFIANZA,
        },
        "resumen": resumir(resultados),
        "resultados": resultados,
    }

    with open(SALIDA, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)

    print()
    print(json.dumps(salida["resumen"], ensure_ascii=False, indent=2))
    print(f"\nDetalle guardado en {SALIDA}")


if __name__ == "__main__":
    main()

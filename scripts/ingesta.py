"""
ingesta.py — Carga la ley en la base vectorial. 📃📁

"""

import os
import re
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)  # para poder importar app/ al correrlo como script

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv(os.path.join(RAIZ, ".env"))

from app.config import ARCHIVO_LEY as ARCHIVO, LEY, CARPETA_CHROMA, COLECCION

# Cohere admite hasta 96 textos por llamada. Usamos 90 por margen. ✋
LOTE = 90

# Y hasta 512 tokens por texto (~2.000 caracteres). Si un artículo es
# más largo, Cohere lo corta y se pierde el final del texto.
MAX_CHUNK = 1500
SOLAPE = 150


# ---------------------------------------------------------------------------
# ✍📃 1. Parseo: convertir el texto plano en artículos ✍📃
# ---------------------------------------------------------------------------

# "Artículo 1° — Fuentes", "Art. 2° — Ambito", "Art. 92 bis. — Período"
PATRON_ARTICULO = re.compile(
    r"^(?:Art\.|Artículo|ARTICULO)\s*(\d+)\s*(bis|ter|BIS|TER)?\s*[.°º]*\s*[—–-]?",
    re.IGNORECASE,
)

PATRON_TITULO = re.compile(r"^TITULO\s+([IVXL]+)\s*$", re.IGNORECASE)
PATRON_CAPITULO = re.compile(r"^CAP[IÍ]TULO\s+([IVXL]+)", re.IGNORECASE)


def _es_encabezado(linea):
    """True si la línea abre un artículo, un título o un capítulo."""
    return bool(
        PATRON_ARTICULO.match(linea)
        or PATRON_TITULO.match(linea)
        or PATRON_CAPITULO.match(linea)
    )


def parsear(texto):
    """
    Recorre el texto y devuelve una lista de artículos. ✍📃

    Cada artículo es un diccionario con su número, su texto completo
    y a qué título y capítulo pertenece. ✍📃

    La clave esta en que el texto del artículo incluye SIEMPRE su nota
    de vigencia final —"(Artículo sustituido por...)" o "(Artículo
    derogado)"—, porque acumulamos hasta el siguiente encabezado.
    """
    lineas = texto.split("\n")

    articulos = []
    actual = None

    titulo_num = titulo_nombre = ""
    capitulo_num = capitulo_nombre = ""

    i = 0
    while i < len(lineas):
        linea = lineas[i].strip()

        # --- ¿Es un TITULO? El nombre viene en la línea siguiente ---
        m = PATRON_TITULO.match(linea)
        if m:
            titulo_num = m.group(1).upper()
            titulo_nombre = ""
            if i + 1 < len(lineas) and not _es_encabezado(lineas[i + 1].strip()):
                titulo_nombre = lineas[i + 1].strip()
                i += 1
            # Un título nuevo reinicia la numeración de capítulos
            capitulo_num = capitulo_nombre = ""
            i += 1
            continue

        # --- ¿Es un CAPITULO? Mismo criterio ---
        m = PATRON_CAPITULO.match(linea)
        if m:
            capitulo_num = m.group(1).upper()
            capitulo_nombre = ""
            if i + 1 < len(lineas) and not _es_encabezado(lineas[i + 1].strip()):
                capitulo_nombre = lineas[i + 1].strip()
                i += 1
            i += 1
            continue

        # --- ¿Es el comienzo de un artículo? ---
        m = PATRON_ARTICULO.match(linea)
        if m:
            if actual:
                articulos.append(actual)

            numero = int(m.group(1))
            sufijo = (m.group(2) or "").lower()
            etiqueta = f"{numero} {sufijo}".strip()

            actual = {
                "numero": numero,
                "sufijo": sufijo,
                "articulo": etiqueta,
                "titulo": titulo_num,
                "titulo_nombre": titulo_nombre,
                "capitulo": capitulo_num,
                "capitulo_nombre": capitulo_nombre,
                "lineas": [linea],
            }
            i += 1
            continue

        # --- Línea de contenido: se suma al artículo en curso ---
        if actual and linea:
            actual["lineas"].append(linea)
        i += 1

    if actual:
        articulos.append(actual)

    articulos = _fusionar_repetidos(articulos)

    # Armamos el texto final de cada artículo
    for art in articulos:
        art["texto"] = "\n".join(art.pop("lineas"))
        art["derogado"] = "derogado" in art["texto"][:200].lower()

    return articulos


def _fusionar_repetidos(articulos):
    """
    Une artículos consecutivos con el mismo número.

    Pasa cuando una línea del cuerpo empieza con algo como
    "Artículo 11: Principios..." y el parser la confunde con un
    encabezado nuevo. Como el número es el mismo y vienen pegados,
    se trata del mismo artículo partido en dos. ✍📃🔟
    """
    if not articulos:
        return articulos

    fusionados = [articulos[0]]

    for art in articulos[1:]:
        anterior = fusionados[-1]
        mismo = (art["numero"] == anterior["numero"]
                 and art["sufijo"] == anterior["sufijo"])

        if mismo:
            anterior["lineas"].extend(art["lineas"])
        else:
            fusionados.append(art)

    return fusionados


# ---------------------------------------------------------------------------
# ✍📃 2. Subdivisión de artículos largos ✍📃
# ---------------------------------------------------------------------------

def partir_si_es_largo(articulo):
    """
    Si el artículo supera el límite de Cohere, lo parte en trozos. ✂📃

    Cada trozo arranca con el encabezado del artículo, para que
    conserve el contexto aunque se lea suelto. ✂📃
    """
    texto = articulo["texto"]

    if len(texto) <= MAX_CHUNK:
        return [texto]

    encabezado = texto.split("\n")[0]
    partes = []
    inicio = 0

    while inicio < len(texto):
        fin = inicio + MAX_CHUNK

        if fin < len(texto):
            corte = texto.rfind(" ", inicio, fin)
            if corte > inicio:
                fin = corte

        fragmento = texto[inicio:fin].strip()

        # Del segundo trozo en adelante repetimos el encabezado
        if partes and not fragmento.startswith(encabezado):
            fragmento = f"{encabezado}\n(continuación)\n{fragmento}"

        if fragmento:
            partes.append(fragmento)

        if fin >= len(texto):
            break

        inicio = fin - SOLAPE
        if texto[inicio] != " ":
            siguiente = texto.find(" ", inicio)
            if siguiente != -1:
                inicio = siguiente + 1

    return partes


def armar_chunks(articulos):
    """📃Convierte la lista de artículos en la lista de fragmentos a indexar. 📃"""
    chunks = []

    for art in articulos:
        partes = partir_si_es_largo(art)

        for n, parte in enumerate(partes, start=1):
            # El número correlativo al principio garantiza que el id sea
            # único aunque un artículo aparezca dos veces en el texto
            # (pasa cuando la ley lo menciona al inicio de una línea).
            correlativo = len(chunks)
            etiqueta = art["articulo"].replace(" ", "_")

            chunks.append({
                "id": f"{correlativo:04d}_art_{etiqueta}_p{n}",
                "texto": parte,
                "metadata": {
                    "ley": LEY,
                    "articulo": art["articulo"],
                    "numero": art["numero"],
                    "titulo": art["titulo"],
                    "titulo_nombre": art["titulo_nombre"],
                    "capitulo": art["capitulo"],
                    "capitulo_nombre": art["capitulo_nombre"],
                    "derogado": art["derogado"],
                    "parte": n,
                    "total_partes": len(partes),
                },
            })

    return chunks


# ---------------------------------------------------------------------------
# ✍📃 3. Reporte de verificación ✍📃
# ---------------------------------------------------------------------------

def reportar(articulos, chunks):
    numeros = sorted({a["numero"] for a in articulos})
    con_sufijo = [a["articulo"] for a in articulos if a["sufijo"]]
    derogados = [a["articulo"] for a in articulos if a["derogado"]]
    largos = [c for c in chunks if c["metadata"]["total_partes"] > 1]

    print()
    print("VERIFICACION DEL PARSEO")
    print("=" * 55)
    print(f"Articulos parseados:    {len(articulos)}")
    print(f"Numeros distintos:      {len(numeros)}  (de {numeros[0]} a {numeros[-1]})")
    faltantes = [n for n in range(numeros[0], numeros[-1] + 1) if n not in numeros]
    print(f"Numeros faltantes:      {len(faltantes)} {faltantes[:15] if faltantes else ''}")
    print(f"Articulos bis/ter:      {len(con_sufijo)} -> {con_sufijo[:8]}")
    print(f"Articulos derogados:    {len(derogados)} -> {derogados[:8]}")

    # ✍🎯 Un artículo puede aparecer dos veces si la ley lo menciona al
    # comienzo de una línea. No rompe nada (los ids son únicos), pero
    # conviene saber cuáles son para revisarlos. ✍🎯
    from collections import Counter
    repetidos = [a for a, n in Counter(x["articulo"] for x in articulos).items() if n > 1]
    print(f"Articulos repetidos:    {len(repetidos)} {repetidos if repetidos else ''}")
    print()
    print(f"Fragmentos a indexar:   {len(chunks)}")
    print(f"  de articulos largos:  {len(largos)}")
    largo_max = max(len(c['texto']) for c in chunks)
    print(f"  fragmento mas largo:  {largo_max:,} caracteres")
    print(f"  supera el limite:     {'SI - REVISAR' if largo_max > 2000 else 'NO'}")
    print(f"Llamadas a Cohere:      {(len(chunks) + LOTE - 1) // LOTE}")
    print()

    print("MUESTRA — PRIMER ARTICULO")
    print("-" * 55)
    print(f"metadata: {chunks[0]['metadata']}")
    print(chunks[0]["texto"][:300])
    print()

    print("MUESTRA — ARTICULO 150 (vacaciones)")
    print("-" * 55)
    for c in chunks:
        if c["metadata"]["numero"] == 150:
            print(f"metadata: {c['metadata']}")
            print(c["texto"][:400])
            break
    print()


# ---------------------------------------------------------------------------
# 📁🚀 4. Carga en Chroma 📁🚀
# ---------------------------------------------------------------------------

def cargar_en_chroma(chunks):
    # Se importa acá y no arriba: así --verificar funciona sin clave de Cohere
    from app.infrastructure import llm

    cliente = chromadb.PersistentClient(
        path=CARPETA_CHROMA,
        settings=Settings(anonymized_telemetry=False),
    )

    # Empezamos de cero para no acumular cargas repetidas.
    try:
        cliente.delete_collection(COLECCION)
        print(f"Coleccion '{COLECCION}' anterior eliminada.")
    except Exception:
        pass

    coleccion = cliente.create_collection(
        name=COLECCION,
        metadata={"hnsw:space": "cosine"},
    )

    total = len(chunks)
    print(f"Generando embeddings de {total} fragmentos en lotes de {LOTE} ...")

    for inicio in range(0, total, LOTE):
        lote = chunks[inicio:inicio + LOTE]
        textos = [c["texto"] for c in lote]

        vectores = llm.embeber(textos, tipo="search_document")

        coleccion.add(
            ids=[c["id"] for c in lote],
            embeddings=vectores,
            documents=textos,
            metadatas=[c["metadata"] for c in lote],
        )

        print(f"  {min(inicio + LOTE, total)}/{total}")

    print()
    print(f"Listo. La coleccion tiene {coleccion.count()} fragmentos.")
    print(f"Guardada en: {CARPETA_CHROMA}")


# ---------------------------------------------------------------------------

def main():
    solo_verificar = "--verificar" in sys.argv

    if not os.path.exists(ARCHIVO):
        raise SystemExit(f"No encuentro {ARCHIVO}. Corré primero: python scripts/descargar_ley.py")

    texto = open(ARCHIVO, encoding="utf-8").read()
    print(f"Leidos {len(texto):,} caracteres de {ARCHIVO}")

    articulos = parsear(texto)
    chunks = armar_chunks(articulos)
    reportar(articulos, chunks)

    if solo_verificar:
        print("Modo verificacion: no se llamo a Cohere ni se cargo nada.")
        print("Si el reporte esta bien, corre:  python scripts/ingesta.py")
        return

    cargar_en_chroma(chunks)


if __name__ == "__main__":
    main()
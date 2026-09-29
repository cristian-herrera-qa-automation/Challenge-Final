"""
descargar_ley.py — Descarga y limpia el texto de la Ley 20.744. 🌐📃

Se corre UNA sola vez:

    python scripts/descargar_ley.py

Guarda tres cosas en data/:
  - ley_20744_crudo.html  el HTML original (por si hay que revisar algo)
  - ley_20744.txt         el texto limpio, que es lo que vamos a indexar
  - reporte.txt           estadísticas para verificar que salió bien
  🌐📃
"""

import os
import re

import requests
from bs4 import BeautifulSoup


URL = "https://www.argentina.gob.ar/normativa/nacional/norma-25552/actualizacion"

# data/ en la raíz del proyecto, se corra desde donde se corra
CARPETA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Desde dónde empieza el texto que nos interesa. Todo lo anterior es
# el menú del sitio, migas de pan y datos de publicación. 👍📃

MARCA_INICIO = "REGIMEN DE CONTRATO DE TRABAJO"

# Textos que marcan el final del articulado. Si aparecen, cortamos ahí.
#
# "Antecedentes normativos" es un anexo de InfoLeg con el historial de
# modificaciones ("Artículo X sustituido por art. Y de la Ley Z").
# No es texto de la ley, y si queda indexado una consulta sobre un
# artículo podría recuperar esa línea en vez del artículo real.
MARCAS_FIN = [
    "Antecedentes normativos",
    "Acerca de esta norma",
    "Presidencia de la Nación",
    "Ministerio de Justicia",
]


def descargar(url):
    """Trae el HTML de la página."""
    print(f"Descargando {url} ...")
    respuesta = requests.get(url, timeout=60, headers={
        "User-Agent": "Mozilla/5.0 (proyecto academico RAG)"
    })
    respuesta.raise_for_status()
    respuesta.encoding = "utf-8"
    print(f"  OK — {len(respuesta.text):,} caracteres de HTML")
    return respuesta.text


def html_a_texto(html):
    """
    📃🌐
    Convierte el HTML en texto plano.

    Saca scripts, estilos y navegación, y pone un salto de línea
    entre bloques para no pegar el final de un párrafo con el
    principio del siguiente. 📃🌐
    """
    soup = BeautifulSoup(html, "lxml")

    for etiqueta in soup(["script", "style", "nav", "header", "footer", "form"]):
        etiqueta.decompose()

    return soup.get_text(separator="\n")


def _sin_tildes(texto):
    """Versión del texto sin tildes y en minúsculas, solo para comparar."""
    reemplazos = str.maketrans("áéíóúüñÁÉÍÓÚÜÑ", "aeiouunAEIOUUN")
    return texto.translate(reemplazos).lower()


def recortar(texto):
    """Se queda solo con el articulado, sin el menú ni el pie del sitio."""
    inicio = texto.find(MARCA_INICIO)
    if inicio == -1:
        print("  AVISO: no se encontró la marca de inicio. Se conserva todo.")
    else:
        texto = texto[inicio:]

    # Buscamos las marcas de fin en la mitad final del texto, para no
    # cortar por una coincidencia en medio de los articulos.
    # La comparación ignora tildes y mayúsculas por si el sitio cambia.
    plano = _sin_tildes(texto)
    limite = int(len(texto) * 0.5)

    posiciones = []
    for marca in MARCAS_FIN:
        pos = plano.rfind(_sin_tildes(marca))
        if pos > limite:
            posiciones.append((pos, marca))

    if posiciones:
        pos, marca = min(posiciones)
        descartado = len(texto) - pos
        porcentaje = descartado / len(texto) * 100

        # Red de seguridad: si el corte se llevaría más de un tercio del
        # texto, algo salió mal. Preferimos conservar de más y avisar.
        if porcentaje > 33:
            print(f"  AVISO: el corte en '{marca}' descartaría el "
                  f"{porcentaje:.0f}% del texto. No se corta.")
        else:
            print(f"  Corte en '{marca}': se descartan {descartado:,} "
                  f"caracteres ({porcentaje:.1f}%)")
            texto = texto[:pos]

    return texto


def limpiar(texto):
    """
    Normaliza el texto:
      - saca espacios repetidos y líneas en blanco de más
      - deja un solo salto de línea entre párrafos
    """
    # Espacios raros que trae el HTML
    texto = texto.replace("\xa0", " ").replace("\u200b", "")

    lineas = []
    for linea in texto.split("\n"):
        linea = re.sub(r"[ \t]+", " ", linea).strip()
        if linea:
            lineas.append(linea)

    return "\n".join(lineas)


def contar_articulos(texto):
    """
    Cuenta cuántos artículos detecta.

    La ley los escribe de varias formas: 'Art. 15.', 'Artículo 11.',
    'Art. 92 bis', 'Art. 132 BIS'. El patrón cubre todas.
    """
    patron = r"(?:^|\n)\s*(?:Art\.|Artículo|ARTICULO)\s*(\d+)\s*(bis|BIS|ter|TER)?"
    return re.findall(patron, texto)


def main():
    os.makedirs(CARPETA, exist_ok=True)

    html = descargar(URL)
    with open(f"{CARPETA}/ley_20744_crudo.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Limpiando ...")
    texto = limpiar(recortar(html_a_texto(html)))

    ruta = f"{CARPETA}/ley_20744.txt"
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(texto)

    # --- Reporte de verificación ---
    articulos = contar_articulos(texto)
    numeros = sorted({int(n) for n, _ in articulos})

    reporte = []
    reporte.append("VERIFICACION DEL CORPUS")
    reporte.append("=" * 50)
    reporte.append(f"Archivo:              {ruta}")
    reporte.append(f"Caracteres:           {len(texto):,}")
    reporte.append(f"Minimo requerido:     100,000")
    reporte.append(f"Cumple el minimo:     {'SI' if len(texto) >= 100_000 else 'NO'}")
    reporte.append(f"Palabras:             {len(texto.split()):,}")
    reporte.append(f"Lineas:               {len(texto.splitlines()):,}")
    reporte.append(f"Articulos detectados: {len(articulos)}")
    if numeros:
        reporte.append(f"Rango de articulos:   {numeros[0]} a {numeros[-1]}")
        faltantes = [n for n in range(numeros[0], numeros[-1] + 1) if n not in numeros]
        reporte.append(f"Numeros faltantes:    {len(faltantes)}")
        if faltantes:
            reporte.append(f"  (primeros 20): {faltantes[:20]}")
    reporte.append("")
    reporte.append("PRIMEROS 400 CARACTERES")
    reporte.append("-" * 50)
    reporte.append(texto[:400])
    reporte.append("")
    reporte.append("ULTIMOS 400 CARACTERES")
    reporte.append("-" * 50)
    reporte.append(texto[-400:])

    salida = "\n".join(reporte)
    with open(f"{CARPETA}/reporte.txt", "w", encoding="utf-8") as f:
        f.write(salida)

    print()
    print(salida)


if __name__ == "__main__":
    main()
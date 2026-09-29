"""
✅ test_pruebas.py — Pruebas de punta a punta. 🧪

    pytest -v        solo dice si cada prueba pasó o falló
    pytest -v -s     además muestra la pregunta, las fuentes y la respuesta

Una prueba por cada caso: pregunta válida, consulta sin evidencia
suficiente, recuperación de fuentes, respuesta del LLM, Human in the Loop
y un error controlado.

Usan mi LLM Cohere. Como el modelo no
responde siempre con las mismas palabras, no se compara el texto exacto:
se verifica lo que tiene que ser fijo (estado, artículo, dato clave,
código de error).

Si dos casos usan la misma pregunta, la segunda vez sale del caché y no
gasta llamadas. 🧪✅
"""

import textwrap


# Preguntas de cada caso 🔍🧪
PREGUNTA_VALIDA = "cuantos días de vacaciones me corresponden con 8 años de antigüedad?"
PREGUNTA_SIN_EVIDENCIA = "cual es la capital de Francia?"
PREGUNTA_FUENTES = "durante cuánto tiempo se presume que un despido es por causa del embarazo?"
PREGUNTA_LLM = "cuando se paga el aguinaldo?"
PREGUNTA_HITL = "que dice la ley del trabajo nocturno?"


def preguntar(cliente, pregunta):
    r = cliente.post("/ask", json={"pregunta": pregunta})
    assert r.status_code == 200, r.text
    return r.json()


def mostrar(**campos):
    """
    Imprime lo que devolvió el sistema. Solo se ve con pytest -v -s.
    No hace llamadas: muestra lo que la prueba ya recibió.
    """
    print()
    for nombre, valor in campos.items():
        etiqueta = f"  {nombre.upper():<12} "
        texto = textwrap.fill(str(valor), width=90, initial_indent=etiqueta,
                              subsequent_indent=" " * len(etiqueta))
        print(texto)


def articulos(fuentes):
    """Las fuentes en una línea: "178, 181, 177 (derogado)"."""
    return ", ".join(f["articulo"] + (" (derogado)" if f["derogado"] else "")
                     for f in fuentes) or "(ninguna)"


# ---------------------------------------------------------------------------
# 1. Pregunta válida 🧪✅
# ---------------------------------------------------------------------------

def test_1_pregunta_valida(cliente):
    datos = preguntar(cliente, PREGUNTA_VALIDA)
    mostrar(pregunta=PREGUNTA_VALIDA, estado=datos["estado"],
            fundamentada=datos["grounded"], respuesta=datos["respuesta"],
            aviso=datos["aviso"])

    assert datos["estado"] == "respondida"
    assert datos["grounded"] is True
    assert datos["respuesta"]
    assert datos["aviso"]  # el aviso legal va siempre


# ---------------------------------------------------------------------------
# 2. Consulta sin evidencia suficiente 🧪✋📃
# ---------------------------------------------------------------------------

def test_2_consulta_sin_evidencia_suficiente(cliente):
    from app.api.routes import FUERA_DE_TEMA

    datos = preguntar(cliente, PREGUNTA_SIN_EVIDENCIA)
    mostrar(pregunta=PREGUNTA_SIN_EVIDENCIA, fuentes=articulos(datos["fuentes"]),
            respuesta=datos["respuesta"])

    # No inventa: avisa que no corresponde a la ley y no cita nada
    assert datos["respuesta"] == FUERA_DE_TEMA
    assert datos["fuentes"] == []
    assert datos["grounded"] is False


# ---------------------------------------------------------------------------
# 3. Recuperación de fuentes 🧪✅
# ---------------------------------------------------------------------------

def test_3_recuperacion_de_fuentes(cliente):
    datos = preguntar(cliente, PREGUNTA_FUENTES)
    mostrar(pregunta=PREGUNTA_FUENTES, fuentes=articulos(datos["fuentes"]))
    for f in datos["fuentes"]:
        print(f"    art. {f['articulo']}: {f['titulo']} / {f['capitulo']} "
              f"(similitud {f['score']}, rerank {f['score_rerank']})")

    numeros = [f["articulo"] for f in datos["fuentes"]]
    assert "178" in numeros  # el artículo del despido por embarazo

    fuente = datos["fuentes"][numeros.index("178")]
    assert fuente["titulo"]
    assert fuente["capitulo"]
    assert fuente["derogado"] is False
    assert 0 < fuente["score"] <= 1


# ---------------------------------------------------------------------------
# 4. Respuesta del LLM 🤖✅
# ---------------------------------------------------------------------------

def test_4_respuesta_del_llm(cliente):
    datos = preguntar(cliente, PREGUNTA_LLM)
    mostrar(pregunta=PREGUNTA_LLM, fuentes=articulos(datos["fuentes"]),
            respuesta=datos["respuesta"])

    respuesta = datos["respuesta"].lower()
    assert "30 de junio" in respuesta      # las dos fechas de pago
    assert "18 de diciembre" in respuesta
    assert "122" in respuesta              # cita el artículo en el que se basa


# ---------------------------------------------------------------------------
# 5. Human in the Loop 🧪✅👨‍💻
# ---------------------------------------------------------------------------

def test_5_human_in_the_loop(cliente):
    from app.api.routes import EN_REVISION

    # La IA genera la respuesta pero no la entrega: entre las fuentes hay
    # un artículo derogado (el 173) y tiene que revisarla una persona.
    datos = preguntar(cliente, PREGUNTA_HITL)
    mostrar(pregunta=PREGUNTA_HITL, estado=datos["estado"],
            motivos=datos["motivos_revision"], fuentes=articulos(datos["fuentes"]),
            usuario_ve=datos["respuesta"])

    assert datos["estado"] == "pending_approval"
    assert "cita_articulo_derogado" in datos["motivos_revision"]
    assert datos["respuesta"] == EN_REVISION  # el usuario no ve el texto
    id_revision = datos["id_revision"]

    # El revisor sí ve la respuesta propuesta
    pendientes = cliente.get("/revisiones/pendientes").json()
    propuesta = next((p for p in pendientes if p["id"] == id_revision), None)
    if propuesta:
        mostrar(revisor_ve=propuesta["propuesta"]["respuesta"])
    assert id_revision in [p["id"] for p in pendientes]

    # El revisor aprueba y la respuesta se entrega
    r = cliente.post(f"/revisiones/{id_revision}/aprobar",
                     json={"revisor": "Prueba automática"})
    assert r.status_code == 200
    resultado = r.json()
    mostrar(decision=f"{resultado['estado']} por {resultado['revisor']}",
            usuario_ve=(resultado["respuesta"] or {}).get("respuesta"))
    assert resultado["estado"] == "aprobada"
    assert resultado["respuesta"]["respuesta"]


# ---------------------------------------------------------------------------
# 6. Error controlado 🚨✋
# ---------------------------------------------------------------------------

def test_6_error_controlado(cliente):
    r = cliente.post("/ask", json={"pregunta": "   "})
    mostrar(pregunta='"   " (vacía)', codigo=r.status_code, respuesta=r.json())

    # Error claro, con el mismo formato que el resto, sin romper la API
    assert r.status_code == 422
    assert r.json() == {"error": "pregunta: La pregunta no puede estar vacía"}

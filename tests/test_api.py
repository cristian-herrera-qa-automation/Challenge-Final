"""
test_api.py — Pruebas de punta a punta de la API.

Cubren los casos que pide la consigna (punto 9):
  1. Pregunta válida
  2. Consulta sin evidencia suficiente
  3. Recuperación de fuentes
  4. Respuesta del LLM
  5. Human in the Loop
  6. Error controlado

Cohere y la base vectorial están reemplazados (ver conftest.py): cada
prueba arma su escenario y verifica qué hace NUESTRO código con él.

    pytest -v
"""

import main
import rag
from conftest import articulo

RESPUESTA_VACACIONES = "Según el artículo 150, te corresponden 21 días corridos de vacaciones."


def escenario_vacaciones(estado):
    """Una consulta normal, con evidencia clara."""
    estado.candidatos = [
        articulo("150", 0.66, texto="Art. 150. — Licencia ordinaria. 21 días corridos..."),
        articulo("153", 0.55),
        articulo("194", 0.50),
    ]
    estado.scores_rerank = {"150": 0.87, "153": 0.63, "194": 0.57}
    estado.respuestas_chat = [RESPUESTA_VACACIONES]


def preguntar(cliente, pregunta="¿Cuántos días de vacaciones me corresponden?"):
    return cliente.post("/ask", json={"pregunta": pregunta})


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_informa_el_estado(cliente):
    r = cliente.get("/health")

    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["fragmentos_indexados"] == 317
    assert r.json()["revisiones_pendientes"] == 0


# ---------------------------------------------------------------------------
# 1 y 4. Pregunta válida -> respuesta del LLM
# ---------------------------------------------------------------------------

def test_pregunta_valida_devuelve_la_respuesta_del_llm(cliente, estado):
    escenario_vacaciones(estado)

    r = preguntar(cliente)

    assert r.status_code == 200
    datos = r.json()
    assert datos["estado"] == "respondida"
    assert datos["respuesta"] == RESPUESTA_VACACIONES
    assert datos["grounded"] is True
    assert datos["aviso"]  # el aviso legal va siempre
    assert estado.llamadas["chat"] == 1


def test_la_misma_pregunta_escrita_distinto_sale_del_cache(cliente, estado):
    escenario_vacaciones(estado)

    primera = preguntar(cliente, "¿Cuántos días de vacaciones me corresponden?")
    segunda = preguntar(cliente, "cuantos dias de vacaciones me corresponden")

    assert segunda.json()["desde_cache"] is True
    assert segunda.json()["respuesta"] == primera.json()["respuesta"]
    assert estado.llamadas["chat"] == 1  # la segunda no llamó al modelo


def test_la_respuesta_llega_sin_emojis(cliente, estado):
    escenario_vacaciones(estado)
    estado.respuestas_chat = ["Según el artículo 150, son 21 días 😊✅"]

    r = preguntar(cliente)

    assert r.json()["respuesta"] == "Según el artículo 150, son 21 días"


# ---------------------------------------------------------------------------
# 3. Recuperación de fuentes
# ---------------------------------------------------------------------------

def test_ask_devuelve_las_fuentes_ordenadas_por_el_reranker(cliente, estado):
    escenario_vacaciones(estado)
    # El reranker da vuelta el orden de similitud: el 194 pasa primero
    estado.scores_rerank = {"150": 0.60, "153": 0.40, "194": 0.90}

    fuentes = preguntar(cliente).json()["fuentes"]

    assert [f["articulo"] for f in fuentes] == ["194", "150", "153"]
    assert fuentes[0]["score_rerank"] == 0.90
    assert fuentes[0]["score"] == 0.50  # la similitud original se conserva


def test_retrieve_muestra_los_fragmentos_sin_llamar_al_llm(cliente, estado):
    escenario_vacaciones(estado)

    r = cliente.post("/retrieve", json={"pregunta": "vacaciones", "top_k": 2})

    assert r.status_code == 200
    datos = r.json()
    assert datos["pertinente"] is True
    assert datos["motivo"] == "ok"
    assert len(datos["fragmentos"]) == 2
    assert "Licencia ordinaria" in datos["fragmentos"][0]["texto"]
    assert estado.llamadas["chat"] == 0


def test_retrieve_puede_excluir_derogados_por_metadata(cliente, estado):
    estado.candidatos = [articulo("173", 0.70, derogado=True), articulo("200", 0.65)]
    estado.scores_rerank = {"173": 0.85, "200": 0.80}

    r = cliente.post("/retrieve", json={"pregunta": "trabajo nocturno", "excluir_derogados": True})

    assert estado.ultimo_where == {"derogado": False}
    assert [f["articulo"] for f in r.json()["fragmentos"]] == ["200"]


def test_retrieve_explica_por_que_no_es_pertinente(cliente, estado):
    estado.candidatos = [articulo("132", 0.34)]

    datos = cliente.post("/retrieve", json={"pregunta": "capital de Francia"}).json()

    assert datos["pertinente"] is False
    assert datos["motivo"] == "similitud_bajo_umbral"
    assert datos["fragmentos"]  # igual muestra los candidatos, para inspeccionarlos


# ---------------------------------------------------------------------------
# 2. Consulta sin evidencia suficiente
# ---------------------------------------------------------------------------

def test_fuera_de_tema_corta_antes_del_reranker_y_del_llm(cliente, estado):
    estado.candidatos = [articulo("132", 0.34), articulo("162", 0.33)]

    datos = preguntar(cliente, "¿Cuál es la capital de Francia?").json()

    assert datos["respuesta"] == main.FUERA_DE_TEMA
    assert datos["grounded"] is False
    assert datos["fuentes"] == []
    assert estado.llamadas["rerank"] == 0
    assert estado.llamadas["chat"] == 0


def test_si_el_reranker_descarta_todo_no_se_llama_al_llm(cliente, estado):
    estado.candidatos = [articulo("27", 0.51), articulo("103", 0.50)]
    estado.scores_rerank = {"27": 0.28, "103": 0.25}  # bajo el umbral de 0.30

    datos = preguntar(cliente, "trabajo como monotributista").json()

    assert datos["respuesta"] == main.FUERA_DE_TEMA
    assert estado.llamadas["chat"] == 0


def test_el_modelo_se_niega_si_el_contexto_no_alcanza(cliente, estado):
    # Caso real: el art. 116 define el salario mínimo pero no dice el monto
    estado.candidatos = [articulo("116", 0.69)]
    estado.scores_rerank = {"116": 0.64}
    estado.respuestas_chat = [rag.SIN_CONTEXTO]

    datos = preguntar(cliente, "¿Cuál es el monto del salario mínimo?").json()

    assert datos["respuesta"] == rag.SIN_CONTEXTO
    assert datos["grounded"] is False
    assert datos["estado"] == "respondida"  # negarse no requiere revisión


# ---------------------------------------------------------------------------
# 5. Human in the Loop
# ---------------------------------------------------------------------------

RESPUESTA_TICKETS = "Según el artículo 131, no te pueden pagar con tickets."


def escenario_confianza_baja(estado):
    estado.candidatos = [articulo("131", 0.51), articulo("106", 0.49)]
    estado.scores_rerank = {"131": 0.48, "106": 0.37}  # mejor fuente < 0.50
    estado.respuestas_chat = [RESPUESTA_TICKETS]


def test_confianza_baja_queda_pendiente_y_no_muestra_el_texto(cliente, estado):
    escenario_confianza_baja(estado)

    datos = preguntar(cliente, "¿Me pueden pagar con tickets?").json()

    assert datos["estado"] == "pending_approval"
    assert datos["motivos_revision"] == ["confianza_baja"]
    assert datos["id_revision"]
    assert RESPUESTA_TICKETS not in datos["respuesta"]
    assert main.cache == {}  # lo pendiente no entra al caché


def test_articulo_derogado_entre_las_fuentes_queda_pendiente(cliente, estado):
    estado.candidatos = [articulo("190", 0.71), articulo("200", 0.70),
                         articulo("173", 0.69, derogado=True)]
    estado.scores_rerank = {"190": 0.84, "200": 0.83, "173": 0.70}
    estado.respuestas_chat = ["Según el artículo 200, la jornada nocturna es de 7 horas."]

    datos = preguntar(cliente, "¿Qué dice la ley del trabajo nocturno?").json()

    assert datos["estado"] == "pending_approval"
    assert datos["motivos_revision"] == ["cita_articulo_derogado"]


def test_si_el_reranker_se_cae_se_responde_pero_con_revision(cliente, estado):
    escenario_vacaciones(estado)
    estado.rerank_caido = True

    datos = preguntar(cliente).json()

    assert datos["estado"] == "pending_approval"
    assert datos["motivos_revision"] == ["confianza_no_medida"]


def test_el_revisor_ve_la_propuesta_y_el_usuario_no(cliente, estado):
    escenario_confianza_baja(estado)
    id_revision = preguntar(cliente).json()["id_revision"]

    cola = cliente.get("/revisiones/pendientes").json()
    usuario = cliente.get(f"/revisiones/{id_revision}").json()

    assert cola[0]["id"] == id_revision
    assert cola[0]["propuesta"]["respuesta"] == RESPUESTA_TICKETS
    assert usuario["estado"] == "pendiente"
    assert usuario["respuesta"] is None


def test_preguntar_de_nuevo_algo_pendiente_no_crea_otra_revision(cliente, estado):
    escenario_confianza_baja(estado)

    primera = preguntar(cliente, "¿Me pueden pagar con tickets?").json()
    segunda = preguntar(cliente, "me pueden pagar con tickets").json()

    assert segunda["id_revision"] == primera["id_revision"]
    assert len(cliente.get("/revisiones/pendientes").json()) == 1
    assert estado.llamadas["chat"] == 1


def test_aprobar_entrega_la_respuesta_y_la_guarda_en_el_cache(cliente, estado):
    escenario_confianza_baja(estado)
    id_revision = preguntar(cliente).json()["id_revision"]

    r = cliente.post(f"/revisiones/{id_revision}/aprobar",
                     json={"revisor": "Dra. Pérez", "comentario": "Correcto."})

    assert r.status_code == 200
    assert r.json()["estado"] == "aprobada"
    assert r.json()["revisor"] == "Dra. Pérez"
    assert r.json()["respuesta"]["respuesta"] == RESPUESTA_TICKETS

    # La próxima vez se entrega directo, sin volver al modelo
    otra = preguntar(cliente).json()
    assert otra["estado"] == "respondida"
    assert otra["desde_cache"] is True
    assert estado.llamadas["chat"] == 1


def test_rechazar_detiene_la_respuesta_y_no_se_vuelve_a_generar(cliente, estado):
    escenario_confianza_baja(estado)
    id_revision = preguntar(cliente).json()["id_revision"]

    cliente.post(f"/revisiones/{id_revision}/rechazar",
                 json={"revisor": "Dra. Pérez", "comentario": "Cita el artículo equivocado."})
    otra = preguntar(cliente).json()

    assert otra["estado"] == "rechazada"
    assert otra["respuesta"] == main.RECHAZADA
    assert RESPUESTA_TICKETS not in str(otra)
    assert estado.llamadas["chat"] == 1


# ---------------------------------------------------------------------------
# 6. Errores controlados
# ---------------------------------------------------------------------------

def test_pregunta_vacia_da_422_con_mensaje_claro(cliente):
    r = cliente.post("/ask", json={"pregunta": "   "})

    assert r.status_code == 422
    assert r.json() == {"error": "pregunta: La pregunta no puede estar vacía"}


def test_json_mal_formado_da_422(cliente):
    r = cliente.post("/ask", content=b'{"pregunta": ', headers={"Content-Type": "application/json"})

    assert r.status_code == 422
    assert "error" in r.json()


def test_top_k_fuera_de_rango_da_422(cliente):
    r = cliente.post("/retrieve", json={"pregunta": "vacaciones", "top_k": 50})

    assert r.status_code == 422
    assert r.json()["error"].startswith("top_k:")


def test_si_cohere_no_responde_da_503_sin_detalles_internos(cliente, estado):
    escenario_vacaciones(estado)
    estado.embed_caido = True

    r = preguntar(cliente)

    assert r.status_code == 503
    assert r.json() == {"error": main.SERVICIO_CAIDO}
    assert main.cache == {}


def test_salida_degenerada_se_reintenta(cliente, estado):
    escenario_vacaciones(estado)
    # Caso real: el modelo devolvió miles de "3" en vez de la respuesta
    estado.respuestas_chat = ["3" * 400, RESPUESTA_VACACIONES]

    datos = preguntar(cliente).json()

    assert datos["respuesta"] == RESPUESTA_VACACIONES
    assert estado.llamadas["chat"] == 2


def test_salida_degenerada_en_todos_los_intentos_da_503_y_no_se_guarda(cliente, estado):
    escenario_vacaciones(estado)
    estado.respuestas_chat = ["3" * 400]

    r = preguntar(cliente)

    assert r.status_code == 503
    assert estado.llamadas["chat"] == 3
    assert main.cache == {}


def test_lenguaje_inapropiado_se_bloquea_antes_de_buscar(cliente, estado):
    datos = preguntar(cliente, "sos un idiota").json()

    assert datos["respuesta"] == main.CONSULTA_BLOQUEADA
    assert estado.llamadas["embed"] == 0


def test_revision_inexistente_da_404(cliente):
    r = cliente.get("/revisiones/noexiste")

    assert r.status_code == 404
    assert r.json() == {"error": "No existe esa revisión"}


def test_resolver_dos_veces_la_misma_revision_da_409(cliente, estado):
    escenario_confianza_baja(estado)
    id_revision = preguntar(cliente).json()["id_revision"]
    cliente.post(f"/revisiones/{id_revision}/aprobar", json={"revisor": "A"})

    r = cliente.post(f"/revisiones/{id_revision}/rechazar", json={"revisor": "B"})

    assert r.status_code == 409


def test_aprobar_sin_nombre_de_revisor_da_422(cliente, estado):
    escenario_confianza_baja(estado)
    id_revision = preguntar(cliente).json()["id_revision"]

    r = cliente.post(f"/revisiones/{id_revision}/aprobar", json={})

    assert r.status_code == 422
    assert r.json()["error"].startswith("revisor:")

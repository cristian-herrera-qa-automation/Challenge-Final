"""
🌐 gui.py — Interfaz para mi asistente conversacional. 🌐

"""

import requests
import streamlit as st

URL_API = "http://localhost:8000/ask"


st.set_page_config(page_title="Asistente Ley 20.744", page_icon="⚖️")

st.title("⚖️ Asistente de Consulta Laboral")
st.caption("Ley de Contrato de Trabajo N° 20.744 (Argentina)")

# Dentro de un form, apretar ENTER en el campo manda la consulta igual que el botón.
with st.form("consulta"):
    pregunta = st.text_input(
        "Escribí tu consulta:",
        placeholder="Ej: ¿cuántos días de vacaciones me corresponden con 8 años de antigüedad?",
    )
    consultar = st.form_submit_button("Consultar", type="primary")


def preguntar_a_la_api(pregunta):
    """Le manda la pregunta a la API y devuelve la respuesta ya decodificada."""
    respuesta = requests.post(URL_API, json={"pregunta": pregunta}, timeout=30)
    respuesta.raise_for_status()
    return respuesta.json()


# Los motivos de revisión que manda la API, dichos para una persona.
MOTIVOS = {
    "cita_articulo_derogado": "la respuesta usa un artículo derogado",
    "confianza_baja": "la evidencia encontrada es débil",
    "confianza_no_medida": "no se pudo medir qué tan confiable es",
}


def mostrar_resultado(datos):
    """Pinta en pantalla la respuesta de la API."""
    if datos["desde_cache"]:
        st.caption("⚡ Respuesta guardada de una consulta anterior")

    if datos["estado"] == "pending_approval":
        motivos = [MOTIVOS.get(m, m) for m in datos["motivos_revision"]]
        st.warning(
            "🕒 **Tu consulta quedó en revisión.** Una persona tiene que "
            "confirmar la respuesta antes de mostrarla, porque "
            f"{' y '.join(motivos)}."
        )
        st.caption(f"Número de revisión: {datos['id_revision']}")
    elif datos["estado"] == "rechazada":
        st.error(datos["respuesta"])
    else:
        st.markdown(datos["respuesta"])

    if datos["fuentes"]:
        st.markdown("**Artículos citados:**")
        for art in datos["fuentes"]:
            etiqueta = f"Art. {art['articulo']} — {art['titulo']}"
            if art["derogado"]:
                etiqueta += " ⚠️ DEROGADO"
            st.markdown(f"- {etiqueta}")

    st.caption(datos["aviso"])


if consultar:
    if not pregunta.strip():
        st.warning("Escribí una consulta antes de enviar.")
    else:
        with st.spinner("Consultando la ley..."):
            try:
                datos = preguntar_a_la_api(pregunta)
                mostrar_resultado(datos)
            except requests.exceptions.ConnectionError:
                st.error(
                    "No se pudo conectar con la API. "
                    "¿Está corriendo `uvicorn main:app --reload`?"
                )
            except requests.exceptions.HTTPError as e:
                st.error(f"La API respondió con un error: {e.response.json().get('error', str(e))}")
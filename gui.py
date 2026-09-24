"""
gui.py — Interfaz gráfica para el asistente de consulta laboral.

Esto NO es la API: es una pantalla que le hace pedidos a la API.
Por eso hacen falta DOS terminales para verla funcionar:

    Terminal 1:  uvicorn main:app --reload
    Terminal 2:  streamlit run gui.py

Se abre sola en el navegador, normalmente en http://localhost:8501
"""

import requests
import streamlit as st

URL_API = "http://localhost:8000/ask"


st.set_page_config(page_title="Asistente Ley 20.744", page_icon="⚖️")

st.title("⚖️ Asistente de Consulta Laboral")
st.caption("Ley de Contrato de Trabajo N° 20.744 (Argentina)")

pregunta = st.text_input(
    "Escribí tu consulta:",
    placeholder="Ej: ¿cuántos días de vacaciones me corresponden con 8 años de antigüedad?",
)

consultar = st.button("Consultar", type="primary")


def preguntar_a_la_api(pregunta):
    """Le manda la pregunta a la API y devuelve la respuesta ya decodificada."""
    respuesta = requests.post(URL_API, json={"pregunta": pregunta}, timeout=30)
    respuesta.raise_for_status()
    return respuesta.json()


def mostrar_resultado(datos):
    """Pinta en pantalla la respuesta de la API."""
    if datos["desde_cache"]:
        st.caption("⚡ Respuesta guardada de una consulta anterior")

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
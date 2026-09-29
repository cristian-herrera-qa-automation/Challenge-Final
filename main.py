"""
main.py — 🤖🔑 La API de mi asistente conversacional sobre la Ley de Contrato de Trabajo. 🤖🔑

Acá solo se arma la aplicación: logging, formato de errores y rutas.

    uvicorn main:app --reload        (Swagger en http://localhost:8000/docs)

Cómo está ordenado el proyecto:
    app/api/routes.py          los endpoints
    app/services/              la lógica: RAG, Human in the Loop y caché
    app/infrastructure/        las conexiones: Cohere (llm) y ChromaDB (vector_store)
    app/schemas/               qué datos entran y salen de la API
    app/config.py              modelos, umbrales y rutas
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from dotenv import load_dotenv

load_dotenv()  # antes de importar las rutas: se lee la API key 🔑

from app.api.routes import router


# ---------------------------------------------------------------------------
# 🔑✅ Logging: cada paso de cada consulta queda registrado 🔑✅
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


app = FastAPI(
    title="Asistente de Consulta - Ley de Contrato de Trabajo",
    description="Sistema RAG sobre la Ley 20.744 (Argentina)",
    version="2.0",
)

app.include_router(router)


# Se registra sobre la excepción de Starlette (la base de la de FastAPI)
# para cubrir también los errores que arma el framework: 404, cuerpo
# mal formado, etc.
@app.exception_handler(StarletteHTTPException)
async def formato_de_error(request: Request, exc: StarletteHTTPException):
    """Todos los errores salen como {"error": "..."} y sin detalles internos."""
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(RequestValidationError)
async def error_de_validacion(request: Request, exc: RequestValidationError):
    """Una entrada inválida (ej. pregunta vacía) sale con el mismo formato."""
    primero = exc.errors()[0]
    # loc es ("body", "pregunta") para un campo, o ("body", 13) cuando el
    # JSON está roto (13 = posición del error): solo nombramos campos.
    campo = ".".join(p for p in primero["loc"] if isinstance(p, str) and p != "body")
    mensaje = primero["msg"].removeprefix("Value error, ")
    return JSONResponse(
        status_code=422,
        content={"error": f"{campo}: {mensaje}" if campo else mensaje},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

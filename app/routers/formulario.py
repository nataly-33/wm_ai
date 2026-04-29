import json
import os
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from groq import Groq
from app.models.schemas import FormularioRequest
from app.services.formulario_service import generar_campos_formulario

router = APIRouter()
logger = logging.getLogger(__name__)

_groq_client = None

def _get_groq():
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if api_key:
            _groq_client = Groq(api_key=api_key)
    return _groq_client


class SugerirCampoRequest(BaseModel):
    nombre_campo: str
    tipo_nodo: Optional[str] = ""
    nombre_politica: Optional[str] = ""
    contexto: Optional[str] = ""


@router.post("/generar-formulario")
async def generar_formulario(request: FormularioRequest):
    try:
        campos = generar_campos_formulario(
            f"Tarea: {request.nombreNodo}\nDescripción: {request.descripcion}"
        )
        return {"campos": campos}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/sugerir-campo")
async def sugerir_campo(request: SugerirCampoRequest):
    """
    Sugiere un valor para un campo de formulario basándose en el contexto del trámite.
    Retorna {"sugerencia": "..."} o {"sugerencia": ""} si Groq no está disponible.
    """
    client = _get_groq()
    if client is None:
        return {"sugerencia": ""}

    prompt = (
        f"Eres asistente para formularios de procesos empresariales.\n\n"
        f"Proceso: {request.nombre_politica or 'proceso interno'}\n"
        f"Tarea actual: {request.tipo_nodo or 'tarea'}\n"
        f"Campo a completar: {request.nombre_campo.replace('_', ' ')}\n"
        f"Texto actual del campo: {request.contexto or '(vacío)'}\n\n"
        f"Genera una sugerencia breve y profesional para este campo (máximo 2 oraciones). "
        f"Responde SOLO con el texto sugerido, sin explicaciones ni comillas."
    )

    try:
        respuesta = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150
        )
        sugerencia = respuesta.choices[0].message.content.strip()
        return {"sugerencia": sugerencia}
    except Exception as e:
        logger.warning(f"sugerir-campo: Groq error: {e}")
        return {"sugerencia": ""}

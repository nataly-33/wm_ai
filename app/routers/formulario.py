from fastapi import APIRouter, HTTPException
from app.models.schemas import FormularioRequest
from app.services.formulario_service import generar_campos_formulario

router = APIRouter()


@router.post("/generar-formulario")
async def generar_formulario(request: FormularioRequest):
    try:
        campos = generar_campos_formulario(
            f"Tarea: {request.nombreNodo}\nDescripción: {request.descripcion}"
        )
        return {"campos": campos}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

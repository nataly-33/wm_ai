from fastapi import APIRouter, HTTPException
from app.models.schemas import DiagramaRequest
from app.services.diagrama_service import generar_diagrama_desde_texto

router = APIRouter()


@router.post("/generar-diagrama")
async def generar_diagrama(request: DiagramaRequest):
    try:
        resultado = generar_diagrama_desde_texto(
            prompt=request.prompt,
            departamentos_empresa=[{"id": d.id, "nombre": d.nombre} for d in request.departamentos]
        )
        return resultado
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error en el servicio de IA: {str(e)}")

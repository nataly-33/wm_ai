from fastapi import APIRouter, HTTPException
from app.models.schemas import AnalisisRequest
from app.services.analisis_service import detectar_cuellos_botella

router = APIRouter()


@router.post("/analizar-politica")
async def analizar_politica(request: AnalisisRequest):
    if len(request.metricas) < 2:
        return {
            "mensaje": "Datos insuficientes. Se necesitan al menos 2 nodos con historial.",
            "resultados": []
        }

    try:
        resultados = detectar_cuellos_botella(
            [m.dict() for m in request.metricas]
        )
        return {"politicaId": request.politicaId, "resultados": resultados}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

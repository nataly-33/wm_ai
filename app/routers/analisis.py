from fastapi import APIRouter, HTTPException
from app.models.schemas import AnalisisRequest
from app.services.analisis_service import detectar_cuellos_botella
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analizar-politica")
async def analizar_politica(request: AnalisisRequest):
    logger.info(f"Recibido analisis: politicaId={request.politicaId}, metricas={len(request.metricas)}")

    if len(request.metricas) < 2:
        return {
            "mensaje": "Datos insuficientes. Se necesitan al menos 2 nodos con historial.",
            "resultados": []
        }

    try:
        # metricas comes as List[Any] (plain dicts from Java), no need to call .dict()
        metricas_list = []
        for m in request.metricas:
            if isinstance(m, dict):
                metricas_list.append(m)
            elif hasattr(m, 'dict'):
                metricas_list.append(m.dict())
            else:
                metricas_list.append(dict(m))

        logger.info(f"Procesando {len(metricas_list)} metricas")
        resultados = detectar_cuellos_botella(metricas_list)
        return {"politicaId": request.politicaId, "resultados": resultados}
    except Exception as e:
        logger.error(f"Error en analisis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import APIRouter, HTTPException
from app.models.schemas import DiagramaRequest
from app.services.diagrama_service import generar_diagrama_desde_texto
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/generar-diagrama")
async def generar_diagrama(request: DiagramaRequest):
    texto = request.get_texto()

    if not texto or not texto.strip():
        raise HTTPException(
            status_code=400,
            detail="El campo 'prompt' o 'descripcion' es requerido y no puede estar vacio"
        )

    logger.info(f"Generando diagrama para texto: {texto[:100]}...")
    logger.info(f"Departamentos recibidos: {[d.nombre for d in request.departamentos]}")

    try:
        resultado = generar_diagrama_desde_texto(
            prompt=texto,
            departamentos_empresa=[{"id": d.id, "nombre": d.nombre} for d in request.departamentos]
        )
        logger.info(f"Diagrama generado: {len(resultado.get('nodos', []))} nodos")
        return resultado
    except ValueError as e:
        logger.error(f"Error de validacion: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error inesperado: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

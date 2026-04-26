from pydantic import BaseModel
from typing import Optional, List, Any


class DepartamentoDto(BaseModel):
    id: str
    nombre: str


class DiagramaRequest(BaseModel):
    prompt: Optional[str] = None
    descripcion: Optional[str] = None
    departamentos: List[DepartamentoDto] = []
    politicaId: Optional[str] = None

    def get_texto(self) -> str:
        return self.prompt or self.descripcion or ""


class MetricasNodoDto(BaseModel):
    nodo_id: str
    nombre_nodo: str
    tiempo_promedio_minutos: float
    cantidad_ejecuciones_activas: int
    tasa_rechazo: float
    tiempo_espera_promedio_minutos: float
    varianza_tiempo: float


class AnalisisRequest(BaseModel):
    politicaId: str
    metricas: Optional[List[Any]] = []


class FormularioRequest(BaseModel):
    descripcion: str
    nombreNodo: Optional[str] = ""

from pydantic import BaseModel
from typing import Optional


class DepartamentoDto(BaseModel):
    id: str
    nombre: str


class DiagramaRequest(BaseModel):
    prompt: str
    departamentos: list[DepartamentoDto]
    politicaId: Optional[str] = None


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
    metricas: list[MetricasNodoDto]


class FormularioRequest(BaseModel):
    descripcion: str
    nombreNodo: str

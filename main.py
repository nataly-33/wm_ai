from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import diagrama, analisis, formulario

app = FastAPI(
    title="WorkflowManager IA Service",
    description="Microservicio de Inteligencia Artificial para WorkflowManager",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(diagrama.router, prefix="/ia", tags=["Diagrama"])
app.include_router(analisis.router, prefix="/ia", tags=["Análisis"])
app.include_router(formulario.router, prefix="/ia", tags=["Formulario"])


@app.get("/health")
def health():
    return {"status": "ok", "service": "WorkflowManager IA"}

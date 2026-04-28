# Guía de Desarrollo — wm_ai

Referencia técnica para el microservicio de IA de WorkflowManager. Cubre cómo funciona cada módulo, qué tecnologías usa y cómo modificar o agregar funcionalidades.

---

## Cómo funciona cada módulo

### 1. Generación de diagramas (`diagrama_service.py`)

**Tecnologías:** spaCy + Groq (Llama 3.1 8B)

**Flujo:**
```
Frontend → POST /ia/generar-diagrama
              ↓
         diagrama_service.py
              ↓
    spaCy: extrae entidades del texto
    (verbos, departamentos mencionados, flujos)
              ↓
    Groq: LLM recibe el prompt enriquecido
    con reglas UML estrictas en el SYSTEM_PROMPT
              ↓
    JSON de nodos + transiciones validado
              ↓
    Backend Java lo convierte en nodos/transiciones
    reales en MongoDB
```

**spaCy** (`es_core_news_sm`) analiza el texto en español para identificar:
- Verbos que serán tareas (detectar, verificar, registrar)
- Entidades que podrían ser departamentos
- Estructura del flujo (secuencia, condiciones, paralelismo)

El resultado del análisis NLP se inyecta en el prompt que se envía a **Groq**. El SYSTEM_PROMPT contiene reglas estrictas del estándar UML 2.5 (tipos de nodos, transiciones, swimlanes, fork/join para paralelismo).

**El modelo Groq** es `llama-3.1-8b-instant` con `temperature=0.2` (baja creatividad, alta consistencia) y `response_format={"type": "json_object"}` para forzar JSON válido.

---

### 2. Generación de formularios (`formulario_service.py`)

**Tecnologías:** Solo Groq (sin spaCy)

**Flujo:**
```
Frontend → POST /ia/generar-formulario
              ↓
         formulario_service.py
              ↓
    Groq: recibe nombre + descripción del nodo
    SYSTEM_PROMPT define tipos de campo válidos
    (TEXTO, NUMERO, FECHA, SELECCION, IMAGEN, ARCHIVO)
              ↓
    JSON de campos validado
```

**Regla clave:** Si el nodo es de decisión (Aprobado/Rechazado), el campo `es_campo_prioridad: true` le indica al motor de workflow qué transición tomar automáticamente.

---

### 3. Análisis de cuellos de botella (`analisis_service.py`)

**Tecnologías:** scikit-learn (RandomForest + GradientBoosting + IsolationForest)

**Flujo:**
```
Backend Java → POST /ia/generar-analisis
(métricas de cada nodo desde MongoDB)
                  ↓
         _enriquecer_metricas()
    (toma 5-7 métricas base y genera 10 features)
                  ↓
         scaler.transform()
    (StandardScaler: normaliza las features)
                  ↓
    RandomForest + GradientBoosting
    → probabilidad de cuello de botella
                  ↓
    IsolationForest
    → detección de anomalías estadísticas
                  ↓
    Ensemble: promedio de RF + GB
    Severidad: CRITICA(>70%), ALTA(>50%), MEDIA(>30%), BAJA
                  ↓
    Lista ordenada por probabilidad con sugerencias
```

**Modelo ML:**
- Entrenado con 450,000 muestras sintéticas (75% normal, 20% moderado, 5% severo)
- Las clases se solapan intencionalmente (distribuciones se cruzan)
- 6% de ruido de etiqueta para accuracy realista (~82-91%)
- Versión: `v3_overlap`
- Archivo: `modelo_cuello_botella.pkl` (~85 MB)

**10 features usadas:**
| Feature | Qué mide |
|---------|----------|
| tiempo_promedio_minutos | Duración promedio del nodo |
| cantidad_ejecuciones_activas | Cuántas instancias hay abiertas |
| tasa_rechazo | % de rechazos sobre completados |
| tiempo_espera_promedio_minutos | Tiempo en cola antes de ejecutar |
| varianza_tiempo | Inconsistencia en los tiempos |
| ratio_completado_rechazado | (1-rechazo)/rechazo |
| tiempo_max_minutos | Peor caso registrado |
| tiempo_min_minutos | Mejor caso registrado |
| tendencia_tiempo | ¿Los tiempos están subiendo? |
| carga_relativa | activas / capacidad_referencia |

---

## Variables de entorno

| Variable | Requerida | Descripción |
|----------|-----------|-------------|
| `GROQ_API_KEY` | Sí (para diagramas/formularios) | Clave de Groq Cloud |

---

## Cómo modificar el SYSTEM_PROMPT de diagramas

El prompt de sistema está en `app/services/diagrama_service.py`, en la constante `SYSTEM_PROMPT`. Define:
- Tipos de nodo válidos (INICIO, TAREA, DECISION, FIN, PARALELO)
- Tipos de transición (LINEAL, ALTERNATIVA, PARALELA)
- Reglas de construcción UML
- Ejemplos de respuesta

Si el ingeniero pide cambiar el comportamiento (por ejemplo, agregar un nuevo tipo de nodo), modificar `SYSTEM_PROMPT` y los tipos válidos en `schemas.py`.

---

## Cómo modificar el modelo ML

1. Editar `analisis_service.py`:
   - `generar_datos_sinteticos()` — Cambiar distribuciones de datos
   - `entrenar_y_guardar()` — Cambiar hiperparámetros de los modelos
   - `_enriquecer_metricas()` — Cambiar cómo se calculan las features
   - `FEATURES` — Agregar/quitar features (debe coincidir en datos y predicción)

2. Cambiar el string `"version"` en `modelo_data` (ej: `"v4_nueva"`) para forzar re-entrenamiento

3. Borrar el `.pkl` viejo y ejecutar `python train_model.py`

---

## Agregar un nuevo endpoint

1. Crear el servicio en `app/services/nuevo_service.py`
2. Crear el router en `app/routers/nuevo.py`:
```python
from fastapi import APIRouter
router = APIRouter()

@router.post("/ia/nuevo-endpoint")
async def nuevo_endpoint(request: NuevoRequest):
    resultado = nuevo_service.procesar(request)
    return resultado
```
3. Registrar el router en `main.py`:
```python
from app.routers import nuevo
app.include_router(nuevo.router, prefix="/ia", tags=["Nuevo"])
```

---

## Logs y debugging

```bash
# Ver logs en tiempo real
uvicorn main:app --reload --log-level debug

# El modelo loguea a nivel INFO:
# "Modelo cargado. AUC-ROC: 0.8934"
# "Modelo no encontrado. Entrenando..."
```

---

## Notas sobre Groq

- **Modelo usado:** `llama-3.1-8b-instant` (rápido y gratuito)
- **Rate limit:** ~14,400 requests/día en el plan gratuito
- **Errores comunes:**
  - `RateLimitError` — Demasiadas peticiones en poco tiempo. El servicio lo captura y retorna HTTP 429
  - `APIConnectionError` — Sin internet o Groq caído. El servicio retorna HTTP 503
  - JSON malformado — El LLM a veces genera JSON incorrecto. Se valida y reintenta

Si se necesita un modelo más potente, cambiar `"llama-3.1-8b-instant"` por `"llama-3.3-70b-versatile"` en ambos services (cuesta más tokens).

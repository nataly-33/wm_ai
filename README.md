# wm_ai

> Sistema de Gestión de Trámites y Políticas de Negocio — Microservicio de Inteligencia Artificial

Microservicio Python del sistema **WorkflowManager**. Se encarga de tres funciones de IA: generar diagramas de actividades desde texto, generar formularios dinámicos desde la descripción de una tarea, y detectar cuellos de botella con machine learning.

---

## Stack

| Tecnología | Versión | Uso |
|------------|---------|-----|
| Python | 3.10+ | Lenguaje principal |
| FastAPI | 0.110+ | Framework REST |
| Groq SDK | latest | LLM (Llama 3.1 8B) para diagramas y formularios |
| spaCy | 3.x | NLP en español (análisis semántico del prompt) |
| scikit-learn | 1.4+ | Modelos ML (RandomForest, GradientBoosting, IsolationForest) |
| NumPy / Pandas | latest | Datos sintéticos y features para ML |
| Pydantic | 2.x | Validación de esquemas de entrada/salida |
| Uvicorn | latest | Servidor ASGI |
| python-dotenv | latest | Variables de entorno |

---

## Requisitos previos

```bash
python --version    # Python 3.10+
pip --version       # pip 23+
git --version       # Git (cualquier versión reciente)
```

---

## Instalación y ejecución local

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/wm_ai.git
cd wm_ai
```

### 2. Crear entorno virtual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt

# Instalar modelo de español de spaCy
python -m spacy download es_core_news_sm
```

### 4. Configurar variables de entorno

Crea un archivo `.env` en la raíz:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> Obtén una clave gratuita en [console.groq.com](https://console.groq.com). Sin esta clave, la generación de diagramas y formularios retorna error. El módulo de análisis ML funciona sin ella.

### 5. Entrenar el modelo de ML (primera vez)

```bash
python train_model.py
```

Esto genera el archivo `modelo_cuello_botella.pkl` (~85 MB). Si el archivo ya existe y es compatible con la versión actual (`v3_overlap`), el servidor lo carga automáticamente sin reentrenar.

### 6. Ejecutar el servidor

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

La API estará disponible en:
- `http://localhost:8001/docs` — Swagger interactivo
- `http://localhost:8001/health` — Estado del servicio

---

## Estructura del proyecto

```
wm_ai/
├── main.py                         ← Entrada de FastAPI, registra los 3 routers
├── train_model.py                  ← Script standalone para entrenar el modelo ML
├── modelo_cuello_botella.pkl       ← Modelo entrenado (generado, no se sube a git)
├── requirements.txt
├── Dockerfile
├── .env                            ← Clave de Groq (no se sube a git)
│
└── app/
    ├── models/
    │   └── schemas.py              ← Pydantic: DiagramaRequest, FormularioRequest, etc.
    │
    ├── routers/
    │   ├── diagrama.py             ← POST /ia/generar-diagrama
    │   ├── formulario.py           ← POST /ia/generar-formulario
    │   └── analisis.py             ← POST /ia/generar-analisis
    │
    └── services/
        ├── diagrama_service.py     ← Lógica: spaCy + Groq → JSON de nodos y transiciones
        ├── formulario_service.py   ← Lógica: Groq → JSON de campos del formulario
        └── analisis_service.py     ← Lógica: ML → detección de cuellos de botella
```

---

## Endpoints de la API

### `POST /ia/generar-diagrama`

Genera un diagrama de actividades UML desde una descripción en texto.

**Request:**
```json
{
  "descripcion": "El cliente solicita una inspección técnica...",
  "departamentos": ["Atención al Cliente", "Técnico", "Facturación"]
}
```

**Response:**
```json
{
  "nodos": [
    { "tempId": "n1", "tipo": "INICIO", "nombre": "Inicio", "departamento": null },
    { "tempId": "n2", "tipo": "TAREA", "nombre": "Recibir solicitud", "departamento": "Atención al Cliente" }
  ],
  "transiciones": [
    { "origen": "n1", "destino": "n2", "tipo": "LINEAL", "etiqueta": null }
  ]
}
```

### `POST /ia/generar-formulario`

Genera los campos de un formulario dinámico para una tarea.

**Request:**
```json
{
  "nombreNodo": "Verificar documentación",
  "descripcion": "El funcionario verifica que el cliente tenga todos los documentos requeridos"
}
```

**Response:**
```json
{
  "campos": [
    { "nombre": "resultado_verificacion", "etiqueta": "Resultado", "tipo": "SELECCION", "requerido": true, "es_campo_prioridad": true, "opciones": ["Aprobado", "Rechazado"] },
    { "nombre": "observaciones", "etiqueta": "Observaciones", "tipo": "TEXTO", "requerido": false, "es_campo_prioridad": false, "opciones": [] }
  ]
}
```

### `POST /ia/generar-analisis`

Detecta cuellos de botella en las ejecuciones de los nodos de una política.

**Request:** lista de métricas por nodo (tiempo promedio, ejecuciones activas, tasa de rechazo, etc.)

**Response:** lista de resultados ordenada por probabilidad de cuello de botella, con severidad y sugerencias.

### `GET /health`

```json
{ "status": "ok", "service": "WorkflowManager IA" }
```

---

## Despliegue en producción (Render)

El servicio se despliega automáticamente en Render. Ver `guia_deploy_render.md` en la raíz del repositorio padre.

Variables de entorno requeridas en Render:
- `GROQ_API_KEY` — Clave de la API de Groq

El modelo `.pkl` se sube junto al código si ya fue entrenado. Si no existe, el servidor lo entrena automáticamente al recibir la primera petición de análisis (puede tardar varios minutos).

---

## Notas importantes

- El modelo ML usa la versión `v3_overlap`. Si existe un `.pkl` de una versión anterior, se re-entrena automáticamente.
- El accuracy del modelo es ~82-91% intencionalmente (datos con solapamiento realista). Un AUC de 1.0 indicaría sobreajuste.
- Groq es gratuito con límite de tokens por minuto. Si se excede, la API retorna error 429 y el backend lo maneja con un mensaje de error al usuario.

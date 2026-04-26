FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requerimientos e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalar el modelo de spaCy
RUN pip install https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.7.0/es_core_news_sm-3.7.0-py3-none-any.whl

# Copiar el resto del codigo
COPY . .

# Entrenar el modelo de IA durante el build para que persista en la imagen
# Esto evita que se pierda al reiniciar el contenedor
RUN python -c "from app.services.analisis_service import entrenar_y_guardar; entrenar_y_guardar()"

# Exponer el puerto
EXPOSE 8001

# Comando para arrancar con uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

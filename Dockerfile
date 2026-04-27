FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download es_core_news_sm

COPY . .

# Si el .pkl ya viene en el repo (recomendado) este paso no hace nada.
# Si no existe, lo entrena con 200K muestras (~2 min, ~350MB RAM).
RUN [ -f modelo_cuello_botella.pkl ] && echo "Modelo pre-entrenado detectado, omitiendo entrenamiento." || python train_model.py

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

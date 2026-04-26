import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
from datetime import datetime

MODELO_PATH = "modelo_cuello_botella.pkl"

FEATURES = [
    "tiempo_promedio_minutos",
    "cantidad_ejecuciones_activas",
    "tasa_rechazo",
    "tiempo_espera_promedio_minutos",
    "varianza_tiempo"
]


# ─────────────────────────────────────────────────────────────────────────────
# GENERACIÓN DE DATOS SINTÉTICOS
# ─────────────────────────────────────────────────────────────────────────────

def generar_datos_sinteticos(n_samples: int = 2000) -> pd.DataFrame:
    np.random.seed(42)
    datos = []

    for _ in range(n_samples):
        es_cuello = np.random.choice([0, 1], p=[0.80, 0.20])

        if es_cuello == 0:
            datos.append({
                "tiempo_promedio_minutos": max(5, np.random.normal(45, 20)),
                "cantidad_ejecuciones_activas": np.random.randint(1, 8),
                "tasa_rechazo": np.random.uniform(0.0, 0.10),
                "tiempo_espera_promedio_minutos": max(0, np.random.normal(15, 8)),
                "varianza_tiempo": max(0, np.random.normal(10, 5)),
                "es_cuello_botella": 0
            })
        else:
            datos.append({
                "tiempo_promedio_minutos": max(60, np.random.normal(360, 120)),
                "cantidad_ejecuciones_activas": np.random.randint(15, 60),
                "tasa_rechazo": np.random.uniform(0.20, 0.60),
                "tiempo_espera_promedio_minutos": max(30, np.random.normal(180, 60)),
                "varianza_tiempo": max(20, np.random.normal(150, 50)),
                "es_cuello_botella": 1
            })

    return pd.DataFrame(datos)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO Y GUARDADO DEL MODELO
# ─────────────────────────────────────────────────────────────────────────────

def entrenar_y_guardar():
    print("=" * 60)
    print("ENTRENAMIENTO — Detección de Cuellos de Botella")
    print("WorkflowManager · Ingeniería de Software I")
    print("=" * 60)

    print("\n[1/4] Generando datos sintéticos...")
    df = generar_datos_sinteticos(2000)
    print(f"      Total: {len(df)} muestras")
    print(f"      Cuellos de botella: {df['es_cuello_botella'].sum()} ({df['es_cuello_botella'].mean()*100:.1f}%)")
    print(f"      Nodos normales: {(df['es_cuello_botella']==0).sum()}")

    print("\n[2/4] Dividiendo datos (80% train / 20% test)...")
    X = df[FEATURES].values
    y = df["es_cuello_botella"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n[3/4] Entrenando modelos...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    isolation = IsolationForest(n_estimators=100, contamination=0.2, random_state=42)
    isolation.fit(X_train_scaled)
    print("      IsolationForest entrenado")

    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train_scaled, y_train)
    print("      RandomForestClassifier entrenado")

    print("\n[4/4] Evaluando en datos de prueba...")
    y_pred = rf.predict(X_test_scaled)

    print("\n--- Reporte de Clasificación (RandomForest) ---")
    print(classification_report(y_test, y_pred,
          target_names=["Proceso Normal", "Cuello de Botella"]))

    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
    print(f"Accuracy: {accuracy*100:.1f}%")

    with open(MODELO_PATH, "wb") as f:
        pickle.dump({
            "isolation_forest": isolation,
            "random_forest": rf,
            "scaler": scaler,
            "features": FEATURES,
            "trained_at": datetime.now().isoformat()
        }, f)

    print(f"\nModelo guardado en: {MODELO_PATH}")
    print("   Iniciar servidor: uvicorn main:app --reload --port 8001")


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DEL MODELO (singleton)
# ─────────────────────────────────────────────────────────────────────────────

_modelo_cache = None


def cargar_modelo():
    global _modelo_cache
    if _modelo_cache is not None:
        return _modelo_cache

    if not os.path.exists(MODELO_PATH):
        print(f"Modelo no encontrado en {MODELO_PATH}. Entrenando...")
        entrenar_y_guardar()

    with open(MODELO_PATH, "rb") as f:
        _modelo_cache = pickle.load(f)

    print(f"Modelo cargado (entrenado el {_modelo_cache.get('trained_at', 'desconocido')})")
    return _modelo_cache


# ─────────────────────────────────────────────────────────────────────────────
# PREDICCIÓN
# ─────────────────────────────────────────────────────────────────────────────

def detectar_cuellos_botella(ejecuciones: list[dict]) -> list[dict]:
    modelo = cargar_modelo()
    rf = modelo["random_forest"]
    isolation = modelo["isolation_forest"]
    scaler = modelo["scaler"]

    if not ejecuciones:
        return []

    resultados = []

    for ejec in ejecuciones:
        X = np.array([[
            ejec.get("tiempo_promedio_minutos", 0),
            ejec.get("cantidad_ejecuciones_activas", 0),
            ejec.get("tasa_rechazo", 0),
            ejec.get("tiempo_espera_promedio_minutos", 0),
            ejec.get("varianza_tiempo", 0)
        ]])

        X_scaled = scaler.transform(X)

        prob_cuello = float(rf.predict_proba(X_scaled)[0][1])
        es_anomalia = isolation.predict(X_scaled)[0] == -1

        if prob_cuello > 0.75:
            severidad = "CRITICA"
        elif prob_cuello > 0.55:
            severidad = "ALTA"
        elif prob_cuello > 0.35:
            severidad = "MEDIA"
        else:
            severidad = "BAJA"

        sugerencias = _generar_sugerencias(ejec, prob_cuello, severidad)

        resultados.append({
            "nodo_id": ejec.get("nodo_id"),
            "nombre_nodo": ejec.get("nombre_nodo"),
            "es_cuello_botella": prob_cuello > 0.5,
            "es_anomalia_estadistica": bool(es_anomalia),
            "probabilidad_cuello": round(prob_cuello, 3),
            "severidad": severidad,
            "sugerencias": sugerencias
        })

    return sorted(resultados, key=lambda x: x["probabilidad_cuello"], reverse=True)


def _generar_sugerencias(ejec: dict, prob: float, severidad: str) -> list[str]:
    sugerencias = []

    tiempo = ejec.get("tiempo_promedio_minutos", 0)
    tasa_rechazo = ejec.get("tasa_rechazo", 0)
    cantidad = ejec.get("cantidad_ejecuciones_activas", 0)

    if tiempo > 240:
        horas = round(tiempo / 60, 1)
        sugerencias.append(
            f"Tiempo promedio de {horas}h excede lo recomendado (max. 4h). "
            f"Considere dividir esta tarea o asignar mas personal."
        )

    if tasa_rechazo > 0.20:
        sugerencias.append(
            f"Tasa de rechazo del {round(tasa_rechazo*100)}% es alta. "
            f"Revise los criterios del formulario o capacite a los funcionarios."
        )

    if cantidad > 20:
        sugerencias.append(
            f"Alta carga: {cantidad} ejecuciones activas simultaneas. "
            f"Considere aumentar el personal en este departamento."
        )

    if severidad == "CRITICA":
        sugerencias.append(
            "Este nodo es el principal cuello de botella del proceso. "
            "Intervencion inmediata recomendada."
        )

    return sugerencias if sugerencias else ["El nodo opera dentro de parametros aceptables."]

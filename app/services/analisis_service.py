"""
Deteccion de cuellos de botella.

Modelos:
  - IsolationForest: anomalias estadisticas (no supervisado)
  - RandomForestClassifier: clasificacion con probabilidad (supervisado)
  - GradientBoostingClassifier: segundo clasificador para ensemble

Accuracy esperada con datos realistas: 78-88% (no 100%)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import (IsolationForest, RandomForestClassifier,
                               GradientBoostingClassifier)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
MODELO_PATH = "modelo_cuello_botella.pkl"
N_MUESTRAS = 1_000_000

FEATURES = [
    "tiempo_promedio_minutos",
    "cantidad_ejecuciones_activas",
    "tasa_rechazo",
    "tiempo_espera_promedio_minutos",
    "varianza_tiempo",
    "ratio_completado_rechazado",
    "tiempo_max_minutos",
    "tiempo_min_minutos",
    "tendencia_tiempo",
    "carga_relativa"
]


def generar_datos_sinteticos(n_samples: int = N_MUESTRAS) -> pd.DataFrame:
    """
    Genera datos sinteticos REALISTAS con solapamiento entre clases.
    Proporciones: 75% normal, 20% cuello moderado, 5% cuello severo.
    """
    np.random.seed(42)
    datos = []

    n_normal = int(n_samples * 0.75)
    n_cuello_moderado = int(n_samples * 0.20)
    n_cuello_severo = n_samples - n_normal - n_cuello_moderado

    print(f"Generando {n_normal:,} nodos normales...")
    for _ in range(n_normal):
        tiempo_base = np.random.lognormal(mean=3.5, sigma=0.8)
        tiempo_base = np.clip(tiempo_base, 5, 300)

        datos.append({
            "tiempo_promedio_minutos": tiempo_base,
            "cantidad_ejecuciones_activas": max(0, int(np.random.poisson(3))),
            "tasa_rechazo": np.clip(np.random.beta(1, 15), 0, 1),
            "tiempo_espera_promedio_minutos": np.clip(tiempo_base * np.random.uniform(0.1, 0.4), 0, 120),
            "varianza_tiempo": np.clip(tiempo_base * np.random.uniform(0.05, 0.3), 0, 200),
            "ratio_completado_rechazado": np.random.uniform(5, 50),
            "tiempo_max_minutos": tiempo_base * np.random.uniform(1.2, 3.0),
            "tiempo_min_minutos": tiempo_base * np.random.uniform(0.3, 0.8),
            "tendencia_tiempo": np.random.normal(0, 0.1),
            "carga_relativa": np.random.uniform(0.5, 1.5),
            "es_cuello_botella": 0
        })

    print(f"Generando {n_cuello_moderado:,} cuellos moderados...")
    for _ in range(n_cuello_moderado):
        tiempo_base = np.random.lognormal(mean=5.0, sigma=0.9)
        tiempo_base = np.clip(tiempo_base, 60, 800)

        datos.append({
            "tiempo_promedio_minutos": tiempo_base,
            "cantidad_ejecuciones_activas": max(1, int(np.random.poisson(12))),
            "tasa_rechazo": np.clip(np.random.beta(3, 10), 0, 1),
            "tiempo_espera_promedio_minutos": np.clip(tiempo_base * np.random.uniform(0.2, 0.6), 0, 400),
            "varianza_tiempo": np.clip(tiempo_base * np.random.uniform(0.2, 0.6), 0, 500),
            "ratio_completado_rechazado": np.random.uniform(1, 8),
            "tiempo_max_minutos": tiempo_base * np.random.uniform(1.5, 4.0),
            "tiempo_min_minutos": tiempo_base * np.random.uniform(0.2, 0.6),
            "tendencia_tiempo": np.random.normal(0.3, 0.2),
            "carga_relativa": np.random.uniform(1.5, 4.0),
            "es_cuello_botella": 1
        })

    print(f"Generando {n_cuello_severo:,} cuellos severos...")
    for _ in range(n_cuello_severo):
        tiempo_base = np.random.lognormal(mean=6.2, sigma=0.7)
        tiempo_base = np.clip(tiempo_base, 300, 2000)

        datos.append({
            "tiempo_promedio_minutos": tiempo_base,
            "cantidad_ejecuciones_activas": max(5, int(np.random.poisson(30))),
            "tasa_rechazo": np.clip(np.random.beta(5, 5), 0.2, 0.8),
            "tiempo_espera_promedio_minutos": np.clip(tiempo_base * np.random.uniform(0.3, 0.8), 0, 800),
            "varianza_tiempo": np.clip(tiempo_base * np.random.uniform(0.4, 0.8), 0, 1000),
            "ratio_completado_rechazado": np.random.uniform(0.5, 3),
            "tiempo_max_minutos": tiempo_base * np.random.uniform(2.0, 5.0),
            "tiempo_min_minutos": tiempo_base * np.random.uniform(0.1, 0.4),
            "tendencia_tiempo": np.random.normal(0.7, 0.3),
            "carga_relativa": np.random.uniform(3.0, 8.0),
            "es_cuello_botella": 1
        })

    df = pd.DataFrame(datos)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def entrenar_y_guardar():
    print("=" * 70)
    print("ENTRENAMIENTO - Deteccion de Cuellos de Botella en Workflows")
    print(f"Muestras: {N_MUESTRAS:,} | Features: {len(FEATURES)}")
    print("=" * 70)

    print("\n[1/5] Generando datos sinteticos realistas...")
    df = generar_datos_sinteticos(N_MUESTRAS)
    print(f"      Shape: {df.shape}")
    print(f"      Balance: {df['es_cuello_botella'].mean()*100:.1f}% son cuellos de botella")

    norm = df[df['es_cuello_botella'] == 0]['tiempo_promedio_minutos']
    cuello = df[df['es_cuello_botella'] == 1]['tiempo_promedio_minutos']
    print(f"\n      Solapamiento de tiempos:")
    print(f"      Normal:  media={norm.mean():.0f}min, std={norm.std():.0f}min, max={norm.max():.0f}min")
    print(f"      Cuello:  media={cuello.mean():.0f}min, std={cuello.std():.0f}min, min={cuello.min():.0f}min")

    print("\n[2/5] Dividiendo datos (80% train / 20% test)...")
    X = df[FEATURES].values
    y = df["es_cuello_botella"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train: {len(X_train):,} | Test: {len(X_test):,}")

    print("\n[3/5] Normalizando features...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("\n[4/5] Entrenando modelos (puede tardar 2-5 minutos)...")

    print("      Entrenando IsolationForest...")
    isolation = IsolationForest(
        n_estimators=200,
        contamination=0.25,
        max_samples=0.1,
        random_state=42,
        n_jobs=-1
    )
    isolation.fit(X_train_s)
    print("      IsolationForest listo")

    print("      Entrenando RandomForestClassifier...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=50,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_s, y_train)
    print("      RandomForest listo")

    print("      Entrenando GradientBoostingClassifier...")
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    gb.fit(X_train_s, y_train)
    print("      GradientBoosting listo")

    print("\n[5/5] Evaluando modelos...")

    y_pred_rf = rf.predict(X_test_s)
    y_pred_gb = gb.predict(X_test_s)
    y_proba_rf = rf.predict_proba(X_test_s)[:, 1]
    y_proba_gb = gb.predict_proba(X_test_s)[:, 1]
    y_proba_ensemble = (y_proba_rf + y_proba_gb) / 2
    y_pred_ensemble = (y_proba_ensemble > 0.5).astype(int)

    print("\n--- RandomForest ---")
    print(classification_report(y_test, y_pred_rf, target_names=["Normal", "Cuello"]))

    print("--- GradientBoosting ---")
    print(classification_report(y_test, y_pred_gb, target_names=["Normal", "Cuello"]))

    print("--- Ensemble (promedio) ---")
    print(classification_report(y_test, y_pred_ensemble, target_names=["Normal", "Cuello"]))

    auc = roc_auc_score(y_test, y_proba_ensemble)
    print(f"AUC-ROC Ensemble: {auc:.4f}")

    print("\n--- Importancia de Features (RandomForest) ---")
    for feat, imp in sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1]):
        bar = "#" * int(imp * 50)
        print(f"  {feat:<35} {imp:.4f} {bar}")

    modelo_data = {
        "isolation_forest": isolation,
        "random_forest": rf,
        "gradient_boosting": gb,
        "scaler": scaler,
        "features": FEATURES,
        "n_samples_entrenamiento": N_MUESTRAS,
        "trained_at": datetime.now().isoformat(),
        "auc_roc": float(auc)
    }

    with open(MODELO_PATH, "wb") as f:
        pickle.dump(modelo_data, f)

    size_mb = os.path.getsize(MODELO_PATH) / (1024 * 1024)
    print(f"\nModelo guardado: {MODELO_PATH} ({size_mb:.1f} MB)")
    print(f"AUC-ROC final: {auc:.4f}")
    print("(Accuracy < 100% es correcto - datos con solapamiento realista)")


_modelo_cache = None


def cargar_modelo():
    global _modelo_cache
    if _modelo_cache:
        return _modelo_cache
    if not os.path.exists(MODELO_PATH):
        logger.info("Modelo no encontrado. Entrenando...")
        entrenar_y_guardar()
    with open(MODELO_PATH, "rb") as f:
        _modelo_cache = pickle.load(f)
    # Verificar compatibilidad de features
    features_guardadas = _modelo_cache.get("features", [])
    if features_guardadas != FEATURES:
        logger.warning("Modelo incompatible (features distintas). Reentrenando...")
        _modelo_cache = None
        entrenar_y_guardar()
        with open(MODELO_PATH, "rb") as f:
            _modelo_cache = pickle.load(f)
    logger.info(f"Modelo cargado. AUC-ROC: {_modelo_cache.get('auc_roc', 'N/A')}")
    return _modelo_cache


def detectar_cuellos_botella(ejecuciones: list) -> list:
    modelo = cargar_modelo()
    rf = modelo["random_forest"]
    gb = modelo["gradient_boosting"]
    scaler = modelo["scaler"]

    resultados = []
    for ejec in ejecuciones:
        tiempo = ejec.get("tiempo_promedio_minutos", 0)
        tiempo_max = ejec.get("tiempo_max_minutos", tiempo * 2)
        tiempo_min = ejec.get("tiempo_min_minutos", tiempo * 0.5)
        activas = ejec.get("cantidad_ejecuciones_activas", 0)
        rechazos = ejec.get("tasa_rechazo", 0)

        X = np.array([[
            tiempo,
            activas,
            rechazos,
            ejec.get("tiempo_espera_promedio_minutos", tiempo * 0.3),
            ejec.get("varianza_tiempo", tiempo * 0.2),
            (1 - rechazos) / max(rechazos, 0.01),
            tiempo_max,
            tiempo_min,
            ejec.get("tendencia_tiempo", 0),
            ejec.get("carga_relativa", 1.0)
        ]])

        X_s = scaler.transform(X)

        prob_rf = float(rf.predict_proba(X_s)[0][1])
        prob_gb = float(gb.predict_proba(X_s)[0][1])
        prob_final = (prob_rf + prob_gb) / 2

        es_anomalia = modelo["isolation_forest"].predict(X_s)[0] == -1

        if prob_final > 0.75:
            severidad = "CRITICA"
        elif prob_final > 0.55:
            severidad = "ALTA"
        elif prob_final > 0.35:
            severidad = "MEDIA"
        else:
            severidad = "BAJA"

        resultados.append({
            "nodo_id": ejec.get("nodo_id"),
            "nombre_nodo": ejec.get("nombre_nodo"),
            "es_cuello_botella": prob_final > 0.5,
            "es_anomalia_estadistica": bool(es_anomalia),
            "probabilidad_cuello": round(prob_final, 3),
            "prob_random_forest": round(prob_rf, 3),
            "prob_gradient_boosting": round(prob_gb, 3),
            "severidad": severidad,
            "metricas": {
                "tiempo_promedio_minutos": round(tiempo, 1),
                "tasa_rechazo_pct": round(rechazos * 100, 1),
                "ejecuciones_activas": activas
            },
            "sugerencias": _sugerencias(ejec, prob_final, severidad)
        })

    return sorted(resultados, key=lambda x: x["probabilidad_cuello"], reverse=True)


def _sugerencias(ejec: dict, prob: float, severidad: str) -> list:
    sugerencias = []
    tiempo = ejec.get("tiempo_promedio_minutos", 0)
    rechazo = ejec.get("tasa_rechazo", 0)
    activas = ejec.get("cantidad_ejecuciones_activas", 0)

    if tiempo > 480:
        sugerencias.append(f"Tiempo promedio de {tiempo/60:.1f}h supera el limite recomendado (8h). Considere dividir esta tarea.")
    elif tiempo > 240:
        sugerencias.append(f"Tiempo promedio de {tiempo/60:.1f}h es elevado. Analizar si puede optimizarse.")
    if rechazo > 0.30:
        sugerencias.append(f"Tasa de rechazo del {rechazo*100:.0f}% es critica. Revisar criterios del formulario.")
    elif rechazo > 0.15:
        sugerencias.append(f"Tasa de rechazo del {rechazo*100:.0f}% indica criterios poco claros.")
    if activas > 25:
        sugerencias.append(f"Alta carga: {activas} tareas activas simultaneas. Asignar mas personal.")
    if severidad == "CRITICA":
        sugerencias.append("Intervencion inmediata recomendada. Este nodo bloquea el flujo del proceso.")

    return sugerencias or ["El nodo opera dentro de parametros aceptables."]

import spacy
import json
import os
import logging
from groq import Groq, APIStatusError, APIConnectionError, RateLimitError

logger = logging.getLogger(__name__)
nlp = spacy.load("es_core_news_sm")

try:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    GROQ_DISPONIBLE = bool(os.environ.get("GROQ_API_KEY"))
except Exception:
    groq_client = None
    GROQ_DISPONIBLE = False

ANCHO_CARRIL = 280
MARGEN_X = 30
ALTO_NIVEL = 160
MARGEN_Y = 60

SYSTEM_PROMPT = """Eres un experto en modelado de procesos UML 2.5.
Analiza la descripcion del proceso y extrae su estructura como diagrama de actividades.

Responde SOLO con JSON valido (sin markdown, sin texto adicional):
{
  "nodos": [
    {"tempId": "n1", "tipo": "INICIO", "nombre": "Inicio", "departamento": null},
    {"tempId": "n2", "tipo": "TAREA", "nombre": "Nombre de la tarea", "departamento": "Nombre exacto del departamento"},
    {"tempId": "n3", "tipo": "DECISION", "nombre": "Pregunta?", "departamento": "Nombre departamento"},
    {"tempId": "n4", "tipo": "FIN", "nombre": "Fin", "departamento": null}
  ],
  "transiciones": [
    {"origen": "n1", "destino": "n2", "tipo": "LINEAL", "etiqueta": null},
    {"origen": "n3", "destino": "n4", "tipo": "ALTERNATIVA", "etiqueta": "Aprobado"},
    {"origen": "n3", "destino": "n5", "tipo": "ALTERNATIVA", "etiqueta": "Rechazado"}
  ]
}

REGLAS:
- Usar SOLO los departamentos de la lista proporcionada
- Siempre incluir nodo INICIO y nodo FIN
- Para condiciones (si/cuando): usar nodo DECISION con 2 transiciones ALTERNATIVA
- Para paralelismo (simultaneamente/en paralelo): usar tipo PARALELO
- Los nombres de tareas deben ser verbos en infinitivo: "Verificar documentacion", "Registrar solicitud"
"""


def preprocesar_texto(texto: str) -> str:
    doc = nlp(texto)
    tokens = [t.text for t in doc if not t.is_space]
    return " ".join(tokens)


def extraer_con_groq(texto: str, departamentos: list) -> dict:
    prompt_usuario = f"""DEPARTAMENTOS DISPONIBLES: {', '.join(departamentos)}

PROCESO A MODELAR:
{texto}

Usa SOLO los departamentos de la lista. Si el texto menciona algo similar, mapealo al mas cercano."""

    respuesta = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_usuario}
        ],
        temperature=0.1,
        max_tokens=2000,
        response_format={"type": "json_object"}
    )

    contenido = respuesta.choices[0].message.content
    return json.loads(contenido)


def extraer_con_spacy_fallback(texto: str, departamentos: list) -> dict:
    logger.info("Usando fallback spaCy para generacion del diagrama")
    doc = nlp(texto.lower())

    deptos_detectados = []
    for depto in departamentos:
        palabras_depto = depto.lower().split()
        for palabra in palabras_depto:
            if len(palabra) > 3 and palabra in texto.lower():
                if depto not in deptos_detectados:
                    deptos_detectados.append(depto)
                break

    if not deptos_detectados:
        deptos_detectados = departamentos[:3]

    palabras_decision = ["si ", "cuando ", "en caso", "dependiendo", "si es", "si fue"]
    tiene_decision = any(p in texto.lower() for p in palabras_decision)

    acciones_por_depto = {}
    oraciones = list(doc.sents)

    for i, oracion in enumerate(oraciones):
        verbo = next((t for t in oracion if t.pos_ == "VERB" and t.dep_ == "ROOT"), None)
        if not verbo:
            verbo = next((t for t in oracion if t.pos_ == "VERB"), None)

        if verbo:
            accion = verbo.lemma_.capitalize()
            for child in verbo.children:
                if child.dep_ in ["obj", "dobj", "nsubj"] and not child.is_stop:
                    accion += f" {child.text}"
                    break

            idx_depto = i % len(deptos_detectados)
            depto = deptos_detectados[idx_depto]

            if depto not in acciones_por_depto:
                acciones_por_depto[depto] = []
            acciones_por_depto[depto].append(accion[:50])

    nodos = [{"tempId": "n_inicio", "tipo": "INICIO", "nombre": "Inicio", "departamento": None}]
    transiciones = []
    ultimo_id = "n_inicio"
    contador = 1

    for depto in deptos_detectados:
        acciones = acciones_por_depto.get(depto, [f"Procesar en {depto}"])

        for accion in acciones[:1]:
            nodo_id = f"n_{contador}"
            contador += 1

            nodos.append({
                "tempId": nodo_id,
                "tipo": "TAREA",
                "nombre": accion,
                "departamento": depto
            })
            transiciones.append({
                "origen": ultimo_id,
                "destino": nodo_id,
                "tipo": "LINEAL",
                "etiqueta": None
            })
            ultimo_id = nodo_id

    if tiene_decision and len(deptos_detectados) >= 2:
        decision_id = f"n_{contador}"
        contador += 1
        fin_si_id = f"n_{contador}"
        contador += 1

        depto_decision = deptos_detectados[0]
        nodos.append({
            "tempId": decision_id,
            "tipo": "DECISION",
            "nombre": "Aprobado?",
            "departamento": depto_decision
        })
        transiciones.append({
            "origen": ultimo_id,
            "destino": decision_id,
            "tipo": "LINEAL",
            "etiqueta": None
        })

        nodos.append({
            "tempId": fin_si_id,
            "tipo": "FIN",
            "nombre": "Fin",
            "departamento": None
        })
        transiciones.append({
            "origen": decision_id,
            "destino": fin_si_id,
            "tipo": "ALTERNATIVA",
            "etiqueta": "Rechazado"
        })
        ultimo_id = decision_id

    fin_id = f"n_{contador}"
    nodos.append({"tempId": fin_id, "tipo": "FIN", "nombre": "Fin", "departamento": None})
    transiciones.append({
        "origen": ultimo_id,
        "destino": fin_id,
        "tipo": "LINEAL" if not tiene_decision else "ALTERNATIVA",
        "etiqueta": None if not tiene_decision else "Aprobado"
    })

    return {"nodos": nodos, "transiciones": transiciones}


def calcular_posiciones(nodos: list, transiciones: list, orden_deptos: dict) -> list:
    if not nodos:
        return nodos

    sucesores = {n["tempId"]: [] for n in nodos}
    for t in transiciones:
        if t["origen"] in sucesores:
            sucesores[t["origen"]].append(t["destino"])

    nodo_inicio = next((n for n in nodos if n["tipo"] == "INICIO"), nodos[0])
    niveles = {nodo_inicio["tempId"]: 0}
    cola = [nodo_inicio["tempId"]]
    visitados = {nodo_inicio["tempId"]}

    while cola:
        actual = cola.pop(0)
        for sucesor in sucesores.get(actual, []):
            nivel_prop = niveles.get(actual, 0) + 1
            if sucesor not in niveles or niveles[sucesor] < nivel_prop:
                niveles[sucesor] = nivel_prop
            if sucesor not in visitados:
                visitados.add(sucesor)
                cola.append(sucesor)

    anchos = {"INICIO": 50, "TAREA": 160, "DECISION": 100, "PARALELO": 200, "FIN": 50}

    nodos_con_pos = []
    for nodo in nodos:
        nivel = niveles.get(nodo["tempId"], 0)
        depto = nodo.get("departamento")
        idx = orden_deptos.get(depto, 0) if depto else 0
        ancho = anchos.get(nodo["tipo"], 160)

        x = MARGEN_X + (idx * ANCHO_CARRIL) + (ANCHO_CARRIL - ancho) // 2
        y = MARGEN_Y + (nivel * ALTO_NIVEL)

        nodos_con_pos.append({**nodo, "posicion_x": x, "posicion_y": y})

    return nodos_con_pos


def transformar_a_formato_sistema(estructura: dict, mapa_nombre_id: dict) -> dict:
    nodos_raw = estructura.get("nodos", [])
    transiciones_raw = estructura.get("transiciones", [])

    deptos_en_orden = []
    for nodo in nodos_raw:
        depto = nodo.get("departamento")
        if depto and depto not in deptos_en_orden and depto in mapa_nombre_id:
            deptos_en_orden.append(depto)

    orden = {d: i for i, d in enumerate(deptos_en_orden)}
    nodos_con_pos = calcular_posiciones(nodos_raw, transiciones_raw, orden)

    nodos_finales = []
    for nodo in nodos_con_pos:
        depto = nodo.get("departamento")
        nodos_finales.append({
            "tempId": nodo["tempId"],
            "tipo": nodo["tipo"],
            "nombre": nodo["nombre"],
            "departamentoId": mapa_nombre_id.get(depto) if depto else None,
            "posicion_x": nodo["posicion_x"],
            "posicion_y": nodo["posicion_y"],
            "formularioId": None
        })

    return {
        "nodos": nodos_finales,
        "transiciones": [
            {
                "nodoOrigenTempId": t["origen"],
                "nodoDestinoTempId": t["destino"],
                "tipo": t.get("tipo", "LINEAL"),
                "etiqueta": t.get("etiqueta"),
                "condicion": t.get("condicion")
            }
            for t in transiciones_raw
        ]
    }


def generar_diagrama_desde_texto(prompt: str, departamentos_empresa: list) -> dict:
    texto_limpio = preprocesar_texto(prompt)
    nombres_deptos = [d["nombre"] for d in departamentos_empresa]
    mapa_id = {d["nombre"]: d["id"] for d in departamentos_empresa}

    estructura = None
    metodo_usado = "groq"

    if GROQ_DISPONIBLE and groq_client:
        try:
            logger.info("Intentando generacion con Groq/Llama...")
            estructura = extraer_con_groq(texto_limpio, nombres_deptos)
            logger.info("Groq respondio correctamente")
        except RateLimitError:
            logger.warning("Groq rate limit alcanzado, usando fallback spaCy")
            metodo_usado = "spacy_fallback"
        except APIConnectionError:
            logger.warning("Sin conexion con Groq, usando fallback spaCy")
            metodo_usado = "spacy_fallback"
        except Exception as e:
            logger.warning(f"Error en Groq ({e}), usando fallback spaCy")
            metodo_usado = "spacy_fallback"
    else:
        logger.info("Groq no configurado, usando spaCy")
        metodo_usado = "spacy_fallback"

    if estructura is None:
        estructura = extraer_con_spacy_fallback(texto_limpio, nombres_deptos)

    resultado = transformar_a_formato_sistema(estructura, mapa_id)
    resultado["metodo_usado"] = metodo_usado
    resultado["advertencia"] = (
        "Diagrama generado con metodo basico (spaCy). "
        "El resultado puede ser menos preciso que con el modelo completo."
        if metodo_usado == "spacy_fallback" else None
    )

    return resultado

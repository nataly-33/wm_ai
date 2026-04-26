import spacy
import json
import os
from groq import Groq
from typing import Optional

nlp = spacy.load("es_core_news_sm")

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 1: Preprocesamiento con spaCy
# ─────────────────────────────────────────────────────────────────────────────

def preprocesar_texto(texto: str) -> str:
    doc = nlp(texto)
    tokens_limpios = []
    for token in doc:
        if not token.is_space and len(token.text.strip()) > 0:
            tokens_limpios.append(token.text)
    return " ".join(tokens_limpios)


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 2: Extracción con Llama 3.1 via Groq
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres un experto en modelado de procesos de negocio con UML 2.5.
Tu tarea es analizar la descripción de un proceso en lenguaje natural y extraer
su estructura como un diagrama de actividades UML 2.5.

REGLAS ESTRICTAS:
1. Responde SOLO con un objeto JSON válido, sin texto adicional, sin markdown, sin explicaciones.
2. Los departamentos deben coincidir EXACTAMENTE con los nombres de la lista proporcionada.
3. Si el texto menciona una condición (si/cuando/dependiendo), crear un nodo DECISION.
4. Si el texto menciona paralelismo (simultáneamente/al mismo tiempo/en paralelo), crear FORK y JOIN.
5. Siempre incluir un nodo INICIO al principio y un nodo FIN al final.
6. Las transiciones de DECISION deben tener etiquetas claras (ej: "Aprobado", "Rechazado").

FORMATO DE RESPUESTA:
{
  "departamentos_detectados": ["nombre1", "nombre2"],
  "nodos": [
    {
      "tempId": "n1",
      "tipo": "INICIO",
      "nombre": "Inicio",
      "departamento": "nombre del departamento o null para INICIO/FIN"
    },
    {
      "tempId": "n2",
      "tipo": "TAREA",
      "nombre": "Nombre descriptivo de la tarea",
      "departamento": "Nombre exacto del departamento"
    },
    {
      "tempId": "n3",
      "tipo": "DECISION",
      "nombre": "¿Pregunta de la decisión?",
      "departamento": "Nombre exacto del departamento"
    },
    {
      "tempId": "n4",
      "tipo": "PARALELO",
      "nombre": "Fork",
      "departamento": null
    },
    {
      "tempId": "n5",
      "tipo": "FIN",
      "nombre": "Fin",
      "departamento": "nombre del departamento o null"
    }
  ],
  "transiciones": [
    {
      "origen": "tempId del nodo origen",
      "destino": "tempId del nodo destino",
      "tipo": "LINEAL",
      "etiqueta": null
    },
    {
      "origen": "tempId del DECISION",
      "destino": "tempId del nodo rama si",
      "tipo": "ALTERNATIVA",
      "etiqueta": "Aprobado"
    },
    {
      "origen": "tempId del DECISION",
      "destino": "tempId del nodo rama no",
      "tipo": "ALTERNATIVA",
      "etiqueta": "Rechazado"
    }
  ]
}"""


def extraer_estructura_con_llm(texto: str, departamentos_empresa: list[str]) -> dict:
    prompt_usuario = f"""DEPARTAMENTOS DISPONIBLES EN LA EMPRESA:
{chr(10).join(f"- {d}" for d in departamentos_empresa)}

DESCRIPCIÓN DEL PROCESO:
{texto}

Extrae la estructura del diagrama de actividades UML 2.5 usando SOLO los departamentos
de la lista de arriba. Si el texto menciona un departamento que no está en la lista,
usa el más similar de la lista."""

    try:
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
    except json.JSONDecodeError as e:
        raise ValueError(f"El modelo no retornó JSON válido: {e}")
    except Exception as e:
        raise ValueError(f"Error al llamar al modelo: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 3: Transformar al formato del sistema y calcular posiciones
# ─────────────────────────────────────────────────────────────────────────────

ANCHO_CARRIL = 280
MARGEN_X = 30
ALTO_NIVEL = 160
MARGEN_Y = 60

ANCHOS_NODO = {
    "INICIO": 50,
    "TAREA": 160,
    "DECISION": 100,
    "PARALELO": 200,
    "FIN": 50
}


def calcular_posiciones(nodos: list[dict], transiciones: list[dict],
                        orden_departamentos: dict[str, int]) -> list[dict]:
    sucesores = {n["tempId"]: [] for n in nodos}
    predecesores = {n["tempId"]: [] for n in nodos}

    for t in transiciones:
        sucesores[t["origen"]].append(t["destino"])
        predecesores[t["destino"]].append(t["origen"])

    nodo_inicio = next((n for n in nodos if n["tipo"] == "INICIO"), None)
    if not nodo_inicio:
        raise ValueError("No se encontró nodo INICIO")

    niveles = {nodo_inicio["tempId"]: 0}
    cola = [nodo_inicio["tempId"]]
    visitados = {nodo_inicio["tempId"]}

    while cola:
        actual = cola.pop(0)
        for sucesor in sucesores.get(actual, []):
            nivel_propuesto = niveles[actual] + 1
            if sucesor not in niveles or niveles[sucesor] < nivel_propuesto:
                niveles[sucesor] = nivel_propuesto
            if sucesor not in visitados:
                visitados.add(sucesor)
                cola.append(sucesor)

    nodos_con_posicion = []
    for nodo in nodos:
        temp_id = nodo["tempId"]
        nivel = niveles.get(temp_id, 0)

        depto = nodo.get("departamento")
        idx_carril = orden_departamentos.get(depto, 0) if depto else 0

        ancho_nodo = ANCHOS_NODO.get(nodo["tipo"], 160)
        x_carril_inicio = MARGEN_X + (idx_carril * ANCHO_CARRIL)
        x = x_carril_inicio + (ANCHO_CARRIL - ancho_nodo) // 2
        y = MARGEN_Y + (nivel * ALTO_NIVEL)

        nodos_con_posicion.append({
            **nodo,
            "posicion_x": x,
            "posicion_y": y
        })

    return nodos_con_posicion


def transformar_a_formato_sistema(estructura_llm: dict,
                                   departamentos_mapeados: dict[str, str]) -> dict:
    nodos_llm = estructura_llm.get("nodos", [])
    transiciones_llm = estructura_llm.get("transiciones", [])

    if not nodos_llm:
        raise ValueError("El modelo no generó ningún nodo")

    deptos_en_orden = []
    for nodo in nodos_llm:
        depto = nodo.get("departamento")
        if depto and depto not in deptos_en_orden and depto in departamentos_mapeados:
            deptos_en_orden.append(depto)

    orden_deptos = {depto: idx for idx, depto in enumerate(deptos_en_orden)}

    nodos_con_pos = calcular_posiciones(nodos_llm, transiciones_llm, orden_deptos)

    nodos_finales = []
    for nodo in nodos_con_pos:
        depto_nombre = nodo.get("departamento")
        depto_id = departamentos_mapeados.get(depto_nombre) if depto_nombre else None

        nodos_finales.append({
            "tempId": nodo["tempId"],
            "tipo": nodo["tipo"],
            "nombre": nodo["nombre"],
            "departamentoId": depto_id,
            "posicion_x": nodo["posicion_x"],
            "posicion_y": nodo["posicion_y"],
            "formularioId": None
        })

    transiciones_finales = []
    for t in transiciones_llm:
        transiciones_finales.append({
            "nodoOrigenTempId": t["origen"],
            "nodoDestinoTempId": t["destino"],
            "tipo": t.get("tipo", "LINEAL"),
            "etiqueta": t.get("etiqueta"),
            "condicion": t.get("condicion")
        })

    return {
        "nodos": nodos_finales,
        "transiciones": transiciones_finales,
        "departamentosDetectados": list(departamentos_mapeados.keys())
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def generar_diagrama_desde_texto(
    prompt: str,
    departamentos_empresa: list[dict]
) -> dict:
    texto_limpio = preprocesar_texto(prompt)
    nombres_deptos = [d["nombre"] for d in departamentos_empresa]
    estructura = extraer_estructura_con_llm(texto_limpio, nombres_deptos)
    mapa_nombre_id = {d["nombre"]: d["id"] for d in departamentos_empresa}
    resultado = transformar_a_formato_sistema(estructura, mapa_nombre_id)
    return resultado

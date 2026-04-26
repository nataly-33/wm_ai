import spacy
import json
import os
import logging
import copy
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

# ──────────────────────────────────────────────────────────────────────────────
# Prompt de sistema — reglas estrictas para el LLM
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un experto en modelado de procesos UML 2.5 para diagramas de actividades con swimlanes.
Tu trabajo es analizar la descripción de un proceso de negocio y generar la estructura del diagrama.

Responde SOLO con JSON válido (sin markdown, sin texto adicional, sin ```):
{
  "nodos": [ ... ],
  "transiciones": [ ... ]
}

═══ ESTRUCTURA DE NODOS ═══
Cada nodo tiene: {"tempId": "n1", "tipo": "...", "nombre": "...", "departamento": "..."}

Tipos válidos:
• INICIO — Exactamente 1 por diagrama. departamento: null
• TAREA — Acción concreta. Nombre = verbo + complemento descriptivo y ÚNICO. departamento: OBLIGATORIO
• DECISION — Punto de bifurcación (pregunta sí/no). departamento: OBLIGATORIO (usar el departamento de la tarea inmediatamente anterior)
• FIN — Al menos 1 (puede haber 2-3 si hay ramas de rechazo). departamento: null
• PARALELO — Barra de sincronización para actividades simultáneas. Se necesitan SIEMPRE 2: un FORK (divide) y un JOIN (une). departamento: null

═══ ESTRUCTURA DE TRANSICIONES ═══
Cada transición: {"origen": "n1", "destino": "n2", "tipo": "...", "etiqueta": null}

Tipos de transición:
• LINEAL — Conexión normal de un nodo al siguiente
• ALTERNATIVA — SOLO sale de nodos DECISION. Siempre en pares: "Aprobado" y "Rechazado"
• PARALELA — SOLO conecta nodos PARALELO con sus ramas paralelas

═══ REGLAS DE PARALELO (FORK/JOIN) ═══
Cuando el texto dice "simultáneamente", "en paralelo", "al mismo tiempo" o "ambas":
1. Crear un nodo PARALELO tipo FORK con nombre "Fork paralelo" (departamento: null)
2. Las transiciones del fork a cada rama son tipo PARALELA
3. Crear las tareas paralelas (cada una en su departamento)
4. Crear un nodo PARALELO tipo JOIN con nombre "Join" (departamento: null)
5. Las transiciones de cada rama al join son tipo LINEAL
6. Después del join, el flujo continúa normalmente

═══ REGLAS CRÍTICAS ═══
1. MÁXIMO 15 nodos en total (incluyendo INICIO y FIN)
2. Cada nombre de nodo DEBE SER ÚNICO — NO repetir nombres como "Revisar solicitud" múltiples veces
3. Nombres descriptivos con verbo + objeto: "Verificar documentación del cliente", "Registrar pago del servicio", NO nombres genéricos como "Revisar" o "Procesar"
4. Cada DECISION debe tener EXACTAMENTE 2 transiciones salientes tipo ALTERNATIVA: una con etiqueta "Aprobado" y otra "Rechazado"
5. La rama "Aprobado" continúa el flujo principal. La rama "Rechazado" puede ir a un nodo FIN o a una tarea de notificación → FIN
6. Si el texto no especifica a qué departamento pertenece una acción, INFERIR el departamento más lógico del contexto
7. Usar SOLO departamentos de la lista proporcionada — mapear sinónimos al más cercano
8. El flujo debe ser secuencial y lógico: INICIO → tareas → decisiones → más tareas → FIN
9. NO crear nodos sueltos sin transiciones
10. Toda tarea y decisión DEBE tener un departamento asignado de la lista

═══ EJEMPLO 1: FLUJO CON DECISIÓN ═══
Para "cliente solicita servicio, se verifica documentación, si aprobado se hace inspección, si no se notifica":

{
  "nodos": [
    {"tempId": "n1", "tipo": "INICIO", "nombre": "Inicio", "departamento": null},
    {"tempId": "n2", "tipo": "TAREA", "nombre": "Recibir solicitud del cliente", "departamento": "Atención al Cliente"},
    {"tempId": "n3", "tipo": "TAREA", "nombre": "Verificar documentación", "departamento": "Atención al Cliente"},
    {"tempId": "n4", "tipo": "DECISION", "nombre": "¿Documentación completa?", "departamento": "Atención al Cliente"},
    {"tempId": "n5", "tipo": "TAREA", "nombre": "Realizar inspección técnica", "departamento": "Técnico"},
    {"tempId": "n6", "tipo": "TAREA", "nombre": "Registrar pago del servicio", "departamento": "Facturación"},
    {"tempId": "n7", "tipo": "FIN", "nombre": "Fin proceso aprobado", "departamento": null},
    {"tempId": "n8", "tipo": "TAREA", "nombre": "Notificar rechazo al cliente", "departamento": "Atención al Cliente"},
    {"tempId": "n9", "tipo": "FIN", "nombre": "Fin rechazo", "departamento": null}
  ],
  "transiciones": [
    {"origen": "n1", "destino": "n2", "tipo": "LINEAL", "etiqueta": null},
    {"origen": "n2", "destino": "n3", "tipo": "LINEAL", "etiqueta": null},
    {"origen": "n3", "destino": "n4", "tipo": "LINEAL", "etiqueta": null},
    {"origen": "n4", "destino": "n5", "tipo": "ALTERNATIVA", "etiqueta": "Aprobado"},
    {"origen": "n4", "destino": "n8", "tipo": "ALTERNATIVA", "etiqueta": "Rechazado"},
    {"origen": "n5", "destino": "n6", "tipo": "LINEAL", "etiqueta": null},
    {"origen": "n6", "destino": "n7", "tipo": "LINEAL", "etiqueta": null},
    {"origen": "n8", "destino": "n9", "tipo": "LINEAL", "etiqueta": null}
  ]
}

═══ EJEMPLO 2: FLUJO CON FORK/JOIN PARALELO ═══
Para "se recibe solicitud, luego simultáneamente se verifica deuda y se verifica estado técnico, cuando ambas terminan se decide si se aprueba":

{
  "nodos": [
    {"tempId": "n1", "tipo": "INICIO", "nombre": "Inicio", "departamento": null},
    {"tempId": "n2", "tipo": "TAREA", "nombre": "Recibir solicitud de reconexión", "departamento": "Atención al Cliente"},
    {"tempId": "n3", "tipo": "PARALELO", "nombre": "Fork paralelo", "departamento": null},
    {"tempId": "n4", "tipo": "TAREA", "nombre": "Verificar deuda pendiente", "departamento": "Facturación"},
    {"tempId": "n5", "tipo": "TAREA", "nombre": "Verificar estado técnico", "departamento": "Técnico"},
    {"tempId": "n6", "tipo": "PARALELO", "nombre": "Join", "departamento": null},
    {"tempId": "n7", "tipo": "DECISION", "nombre": "¿Todo aprobado?", "departamento": "Técnico"},
    {"tempId": "n8", "tipo": "TAREA", "nombre": "Ejecutar reconexión", "departamento": "Técnico"},
    {"tempId": "n9", "tipo": "FIN", "nombre": "Fin aprobado", "departamento": null},
    {"tempId": "n10", "tipo": "TAREA", "nombre": "Informar impedimento", "departamento": "Atención al Cliente"},
    {"tempId": "n11", "tipo": "FIN", "nombre": "Fin rechazo", "departamento": null}
  ],
  "transiciones": [
    {"origen": "n1", "destino": "n2", "tipo": "LINEAL", "etiqueta": null},
    {"origen": "n2", "destino": "n3", "tipo": "LINEAL", "etiqueta": null},
    {"origen": "n3", "destino": "n4", "tipo": "PARALELA", "etiqueta": null},
    {"origen": "n3", "destino": "n5", "tipo": "PARALELA", "etiqueta": null},
    {"origen": "n4", "destino": "n6", "tipo": "LINEAL", "etiqueta": null},
    {"origen": "n5", "destino": "n6", "tipo": "LINEAL", "etiqueta": null},
    {"origen": "n6", "destino": "n7", "tipo": "LINEAL", "etiqueta": null},
    {"origen": "n7", "destino": "n8", "tipo": "ALTERNATIVA", "etiqueta": "Aprobado"},
    {"origen": "n7", "destino": "n10", "tipo": "ALTERNATIVA", "etiqueta": "Rechazado"},
    {"origen": "n8", "destino": "n9", "tipo": "LINEAL", "etiqueta": null},
    {"origen": "n10", "destino": "n11", "tipo": "LINEAL", "etiqueta": null}
  ]
}
"""



def preprocesar_texto(texto: str) -> str:
    doc = nlp(texto)
    tokens = [t.text for t in doc if not t.is_space]
    return " ".join(tokens)


def extraer_con_groq(texto: str, departamentos: list) -> dict:
    prompt_usuario = f"""DEPARTAMENTOS DISPONIBLES (usar SOLO estos nombres exactos): {', '.join(departamentos)}

PROCESO A MODELAR:
{texto}

INSTRUCCIONES ADICIONALES:
- Usa SOLO los departamentos de la lista anterior
- Si una acción no menciona departamento, asígnalo al departamento más lógico según el contexto
- Para DECISION, usa el departamento de la tarea que la precede
- Cada nombre de nodo debe ser ÚNICO y descriptivo
- Máximo 12 nodos en total
- Responde SOLO con JSON válido, sin texto adicional"""

    respuesta = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_usuario}
        ],
        temperature=0.1,
        max_tokens=3000,
        response_format={"type": "json_object"}
    )

    contenido = respuesta.choices[0].message.content
    return json.loads(contenido)


def extraer_con_spacy_fallback(texto: str, departamentos: list) -> dict:
    logger.info("Usando fallback spaCy para generacion del diagrama")
    doc = nlp(texto.lower())

    # Detect departments mentioned in text
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

    palabras_decision = ["si ", "cuando ", "en caso", "dependiendo", "si es", "si fue", "aprobad", "rechaz"]
    tiene_decision = any(p in texto.lower() for p in palabras_decision)

    # Extract actions from sentences
    acciones = []
    oraciones = list(doc.sents)
    for oracion in oraciones:
        verbo = next((t for t in oracion if t.pos_ == "VERB" and t.dep_ == "ROOT"), None)
        if not verbo:
            verbo = next((t for t in oracion if t.pos_ == "VERB"), None)
        if verbo:
            # Build action phrase
            complemento = ""
            for child in verbo.children:
                if child.dep_ in ["obj", "dobj", "nsubj", "obl"] and not child.is_stop:
                    complemento = f" {child.text}"
                    break
            accion = f"{verbo.lemma_.capitalize()}{complemento}"
            if accion not in [a[0] for a in acciones]:
                # Try to match to a department
                depto_idx = 0
                for i, depto in enumerate(deptos_detectados):
                    if any(p in oracion.text.lower() for p in depto.lower().split() if len(p) > 3):
                        depto_idx = i
                        break
                acciones.append((accion[:50], deptos_detectados[depto_idx % len(deptos_detectados)]))

    if not acciones:
        acciones = [(f"Procesar en {d}", d) for d in deptos_detectados[:3]]

    # Build diagram structure
    nodos = [{"tempId": "n_inicio", "tipo": "INICIO", "nombre": "Inicio", "departamento": None}]
    transiciones = []
    ultimo_id = "n_inicio"
    contador = 1

    for accion, depto in acciones:
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

    if tiene_decision and len(deptos_detectados) >= 1:
        decision_id = f"n_{contador}"
        contador += 1
        fin_rechazo_id = f"n_{contador}"
        contador += 1
        notif_id = f"n_{contador}"
        contador += 1

        depto_decision = deptos_detectados[0]
        nodos.append({
            "tempId": decision_id,
            "tipo": "DECISION",
            "nombre": "¿Aprobado?",
            "departamento": depto_decision
        })
        transiciones.append({
            "origen": ultimo_id,
            "destino": decision_id,
            "tipo": "LINEAL",
            "etiqueta": None
        })

        # Rejected branch
        nodos.append({
            "tempId": notif_id,
            "tipo": "TAREA",
            "nombre": "Notificar rechazo",
            "departamento": depto_decision
        })
        nodos.append({
            "tempId": fin_rechazo_id,
            "tipo": "FIN",
            "nombre": "Fin rechazo",
            "departamento": None
        })
        transiciones.append({
            "origen": decision_id,
            "destino": notif_id,
            "tipo": "ALTERNATIVA",
            "etiqueta": "Rechazado"
        })
        transiciones.append({
            "origen": notif_id,
            "destino": fin_rechazo_id,
            "tipo": "LINEAL",
            "etiqueta": None
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


# ──────────────────────────────────────────────────────────────────────────────
# Post-procesamiento y validación del output del LLM
# ──────────────────────────────────────────────────────────────────────────────

def validar_y_limpiar_estructura(estructura: dict, departamentos_validos: list) -> dict:
    """
    Validates and cleans the LLM output to ensure:
    - Unique node names
    - Valid department assignments
    - Proper DECISION branching (exactly 2 ALTERNATIVA transitions)
    - Exactly 1 INICIO, at least 1 FIN
    - All transitions reference existing nodes
    - Max 15 nodes
    """
    nodos = estructura.get("nodos", [])
    transiciones = estructura.get("transiciones", [])

    if not nodos:
        logger.warning("LLM returned empty nodos list")
        return estructura

    deptos_lower = {d.lower(): d for d in departamentos_validos}

    # --- 1. Deduplicate node names ---
    seen_names = {}
    unique_nodos = []
    id_remap = {}  # old_id -> new_id for deduplication

    for nodo in nodos:
        nombre = nodo.get("nombre", "").strip()
        temp_id = nodo.get("tempId", "")

        if not nombre:
            nombre = f"Nodo {temp_id}"
            nodo["nombre"] = nombre

        nombre_lower = nombre.lower()

        if nombre_lower in seen_names:
            # Remap this node's ID to the first occurrence
            original_id = seen_names[nombre_lower]
            id_remap[temp_id] = original_id
            logger.info(f"Deduplicando nodo '{nombre}': {temp_id} -> {original_id}")
        else:
            seen_names[nombre_lower] = temp_id
            unique_nodos.append(nodo)

    nodos = unique_nodos

    # --- 2. Remap transitions for deduplicated nodes ---
    for tr in transiciones:
        if tr.get("origen") in id_remap:
            tr["origen"] = id_remap[tr["origen"]]
        if tr.get("destino") in id_remap:
            tr["destino"] = id_remap[tr["destino"]]

    # Remove duplicate transitions after remapping
    seen_transitions = set()
    unique_trans = []
    for tr in transiciones:
        key = (tr.get("origen"), tr.get("destino"), tr.get("etiqueta"))
        if key not in seen_transitions:
            seen_transitions.add(key)
            unique_trans.append(tr)
    transiciones = unique_trans

    # --- 3. Ensure exactly 1 INICIO ---
    inicios = [n for n in nodos if n.get("tipo") == "INICIO"]
    if len(inicios) == 0:
        nodos.insert(0, {"tempId": "n_auto_inicio", "tipo": "INICIO", "nombre": "Inicio", "departamento": None})
        # Connect to first non-INICIO node
        first_other = next((n for n in nodos if n.get("tipo") != "INICIO"), None)
        if first_other:
            transiciones.insert(0, {
                "origen": "n_auto_inicio",
                "destino": first_other["tempId"],
                "tipo": "LINEAL",
                "etiqueta": None
            })
    elif len(inicios) > 1:
        for extra in inicios[1:]:
            nodos.remove(extra)

    # --- 4. Ensure at least 1 FIN ---
    fines = [n for n in nodos if n.get("tipo") == "FIN"]
    if len(fines) == 0:
        nodos.append({"tempId": "n_auto_fin", "tipo": "FIN", "nombre": "Fin", "departamento": None})
        # Find nodes with no outgoing transitions
        nodo_ids = {n["tempId"] for n in nodos}
        nodos_con_salida = {tr["origen"] for tr in transiciones if tr.get("origen") in nodo_ids}
        sin_salida = [n for n in nodos if n["tempId"] not in nodos_con_salida
                      and n.get("tipo") not in ("FIN", "INICIO")]
        if sin_salida:
            transiciones.append({
                "origen": sin_salida[-1]["tempId"],
                "destino": "n_auto_fin",
                "tipo": "LINEAL",
                "etiqueta": None
            })

    # --- 5. Fix department assignments ---
    nodo_by_id = {n["tempId"]: n for n in nodos}
    predecessors = {}
    successors = {}
    for tr in transiciones:
        orig = tr.get("origen")
        dest = tr.get("destino")
        if orig and dest:
            successors.setdefault(orig, []).append(dest)
            predecessors.setdefault(dest, []).append(orig)

    for nodo in nodos:
        if nodo.get("tipo") in ("INICIO", "FIN"):
            nodo["departamento"] = None
            continue

        depto = nodo.get("departamento")
        if depto and depto in departamentos_validos:
            continue  # Valid department

        # Try to match case-insensitively
        if depto:
            matched = deptos_lower.get(depto.lower())
            if matched:
                nodo["departamento"] = matched
                continue

            # Try partial match
            depto_lower = depto.lower()
            for key, val in deptos_lower.items():
                if depto_lower in key or key in depto_lower:
                    nodo["departamento"] = val
                    logger.info(f"Matched dept '{depto}' -> '{val}' by partial match")
                    break
            else:
                depto = None  # No match found

        if not depto:
            # Inherit from predecessor
            preds = predecessors.get(nodo["tempId"], [])
            for pred_id in preds:
                pred = nodo_by_id.get(pred_id)
                if pred and pred.get("departamento") and pred["departamento"] in departamentos_validos:
                    nodo["departamento"] = pred["departamento"]
                    logger.info(f"Inherited dept '{pred['departamento']}' for node '{nodo['nombre']}' from predecessor")
                    break

        if not nodo.get("departamento") or nodo["departamento"] not in departamentos_validos:
            # Fall back to first department
            nodo["departamento"] = departamentos_validos[0] if departamentos_validos else None
            logger.info(f"Fallback dept '{nodo['departamento']}' for node '{nodo['nombre']}'")

    # --- 6. Validate DECISION nodes have exactly 2 ALTERNATIVA outgoing ---
    for nodo in nodos:
        if nodo.get("tipo") != "DECISION":
            continue

        nodo_id = nodo["tempId"]
        salientes = [tr for tr in transiciones if tr.get("origen") == nodo_id]
        alternativas = [tr for tr in salientes if tr.get("tipo") == "ALTERNATIVA"]

        if len(alternativas) == 0:
            # Convert existing outgoing to ALTERNATIVA or create branches
            if len(salientes) >= 2:
                salientes[0]["tipo"] = "ALTERNATIVA"
                salientes[0]["etiqueta"] = "Aprobado"
                salientes[1]["tipo"] = "ALTERNATIVA"
                salientes[1]["etiqueta"] = "Rechazado"
            elif len(salientes) == 1:
                salientes[0]["tipo"] = "ALTERNATIVA"
                salientes[0]["etiqueta"] = "Aprobado"
                # Create rejection branch
                fin_id = f"n_auto_fin_rej_{nodo_id}"
                nodos.append({"tempId": fin_id, "tipo": "FIN", "nombre": f"Fin rechazo", "departamento": None})
                transiciones.append({
                    "origen": nodo_id,
                    "destino": fin_id,
                    "tipo": "ALTERNATIVA",
                    "etiqueta": "Rechazado"
                })
            else:
                # No outgoing at all — create both branches
                fin_ok_id = f"n_auto_fin_ok_{nodo_id}"
                fin_rej_id = f"n_auto_fin_rej_{nodo_id}"
                nodos.append({"tempId": fin_ok_id, "tipo": "FIN", "nombre": "Fin aprobado", "departamento": None})
                nodos.append({"tempId": fin_rej_id, "tipo": "FIN", "nombre": "Fin rechazo", "departamento": None})
                transiciones.append({"origen": nodo_id, "destino": fin_ok_id, "tipo": "ALTERNATIVA", "etiqueta": "Aprobado"})
                transiciones.append({"origen": nodo_id, "destino": fin_rej_id, "tipo": "ALTERNATIVA", "etiqueta": "Rechazado"})
        elif len(alternativas) == 1:
            # Only one ALTERNATIVA — add the missing one
            existing_label = alternativas[0].get("etiqueta", "")
            if "aprob" in (existing_label or "").lower():
                missing_label = "Rechazado"
            else:
                missing_label = "Aprobado"
            fin_id = f"n_auto_fin_alt_{nodo_id}"
            nodos.append({"tempId": fin_id, "tipo": "FIN", "nombre": f"Fin {missing_label.lower()}", "departamento": None})
            transiciones.append({
                "origen": nodo_id,
                "destino": fin_id,
                "tipo": "ALTERNATIVA",
                "etiqueta": missing_label
            })

        # Ensure labels
        salientes = [tr for tr in transiciones if tr.get("origen") == nodo_id and tr.get("tipo") == "ALTERNATIVA"]
        has_aprobado = any(("aprob" in (tr.get("etiqueta") or "").lower()) for tr in salientes)
        has_rechazado = any(("rechaz" in (tr.get("etiqueta") or "").lower()) for tr in salientes)
        if not has_aprobado and len(salientes) >= 1:
            salientes[0]["etiqueta"] = "Aprobado"
        if not has_rechazado and len(salientes) >= 2:
            salientes[1]["etiqueta"] = "Rechazado"

    # --- 7. Remove invalid transitions (references to non-existent nodes) ---
    valid_ids = {n["tempId"] for n in nodos}
    transiciones = [tr for tr in transiciones if tr.get("origen") in valid_ids and tr.get("destino") in valid_ids]

    # --- 8. Remove self-referencing transitions ---
    transiciones = [tr for tr in transiciones if tr.get("origen") != tr.get("destino")]

    # --- 9. Cap at 15 nodes ---
    if len(nodos) > 15:
        logger.warning(f"LLM generated {len(nodos)} nodes, capping to 15")
        # Keep INICIO, FIN, and first non-special nodes
        inicio = [n for n in nodos if n["tipo"] == "INICIO"]
        fines = [n for n in nodos if n["tipo"] == "FIN"]
        otros = [n for n in nodos if n["tipo"] not in ("INICIO", "FIN")]
        nodos = inicio + otros[:13] + fines[:2]
        valid_ids = {n["tempId"] for n in nodos}
        transiciones = [tr for tr in transiciones if tr.get("origen") in valid_ids and tr.get("destino") in valid_ids]

    logger.info(f"Post-procesamiento: {len(nodos)} nodos, {len(transiciones)} transiciones")
    return {"nodos": nodos, "transiciones": transiciones}


# ──────────────────────────────────────────────────────────────────────────────
# Posicionamiento — basic since frontend will re-layout with its superior engine
# ──────────────────────────────────────────────────────────────────────────────

def calcular_posiciones(nodos: list, transiciones: list, orden_deptos: dict) -> list:
    if not nodos:
        return nodos

    # Build successor map for level calculation
    sucesores = {n["tempId"]: [] for n in nodos}
    predecesores = {n["tempId"]: [] for n in nodos}
    for t in transiciones:
        orig = t.get("origen")
        dest = t.get("destino")
        if orig in sucesores and dest in sucesores:
            sucesores[orig].append(dest)
            predecesores[dest].append(orig)

    # Find INICIO
    nodo_inicio = next((n for n in nodos if n["tipo"] == "INICIO"), nodos[0])

    # BFS for topological levels
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

    # Assign level 0 to unvisited nodes
    for n in nodos:
        if n["tempId"] not in niveles:
            niveles[n["tempId"]] = max(niveles.values(), default=0) + 1

    # Align branches: if a DECISION has 2 outgoing, put them at the same level
    for n in nodos:
        if n["tipo"] == "DECISION":
            hijos = sucesores.get(n["tempId"], [])
            if len(hijos) >= 2:
                max_nivel = max(niveles.get(h, 0) for h in hijos)
                for h in hijos:
                    niveles[h] = max_nivel

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
        depto_id = mapa_nombre_id.get(depto) if depto else None

        # Fallback: if department not in map, try partial match
        if depto and not depto_id:
            depto_lower = depto.lower()
            for nombre, id_val in mapa_nombre_id.items():
                if depto_lower in nombre.lower() or nombre.lower() in depto_lower:
                    depto_id = id_val
                    break

        nodos_finales.append({
            "tempId": nodo["tempId"],
            "tipo": nodo["tipo"],
            "nombre": nodo["nombre"],
            "departamentoId": depto_id,
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
            logger.info(f"Groq respondio: {len(estructura.get('nodos', []))} nodos, {len(estructura.get('transiciones', []))} transiciones")

            # Post-process and validate LLM output
            estructura = validar_y_limpiar_estructura(estructura, nombres_deptos)

        except RateLimitError:
            logger.warning("Groq rate limit alcanzado, usando fallback spaCy")
            metodo_usado = "spacy_fallback"
        except APIConnectionError:
            logger.warning("Sin conexion con Groq, usando fallback spaCy")
            metodo_usado = "spacy_fallback"
        except json.JSONDecodeError as e:
            logger.warning(f"Groq devolvio JSON invalido ({e}), usando fallback spaCy")
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

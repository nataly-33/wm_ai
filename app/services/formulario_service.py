import json
import os
from groq import Groq

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

TIPOS_CAMPO_VALIDOS = ["TEXTO", "NUMERO", "FECHA", "SELECCION", "IMAGEN", "ARCHIVO"]

SYSTEM_PROMPT_FORMULARIO = """Eres un experto en diseño de formularios para procesos empresariales.
Dado una descripción de una tarea, genera los campos que el formulario debería tener.

Responde SOLO con JSON válido, sin texto adicional:
{
  "campos": [
    {
      "nombre": "nombre_tecnico_snake_case",
      "etiqueta": "Texto visible para el usuario",
      "tipo": "TEXTO|NUMERO|FECHA|SELECCION|IMAGEN|ARCHIVO",
      "requerido": true|false,
      "es_campo_prioridad": true|false,
      "opciones": ["opcion1", "opcion2"]
    }
  ]
}

REGLAS:
- nombre: snake_case sin tildes, máx 30 chars
- tipo SELECCION: incluir opciones relevantes
- es_campo_prioridad: true SOLO si este campo determinará qué rama seguir después (ej: "Aprobado"/"Rechazado")
- opciones: solo para tipo SELECCION, array de strings
- Para campos de decisión binaria: usar SELECCION con ["Aprobado","Rechazado"] o ["Sí","No"]
"""


def generar_campos_formulario(descripcion_nodo: str) -> list[dict]:
    respuesta = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_FORMULARIO},
            {"role": "user", "content": f"Genera los campos para este formulario:\n{descripcion_nodo}"}
        ],
        temperature=0.2,
        max_tokens=1000,
        response_format={"type": "json_object"}
    )

    data = json.loads(respuesta.choices[0].message.content)
    campos = data.get("campos", [])

    campos_validos = []
    for campo in campos:
        if campo.get("tipo") in TIPOS_CAMPO_VALIDOS:
            campos_validos.append({
                "nombre": campo.get("nombre", "campo"),
                "etiqueta": campo.get("etiqueta", "Campo"),
                "tipo": campo.get("tipo", "TEXTO"),
                "requerido": campo.get("requerido", True),
                "es_campo_prioridad": campo.get("es_campo_prioridad", False),
                "opciones": campo.get("opciones", []) if campo.get("tipo") == "SELECCION" else []
            })

    return campos_validos

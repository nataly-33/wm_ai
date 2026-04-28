# Guía de Contribución — wm_ai

Convenciones y reglas para contribuir al microservicio de IA de WorkflowManager.

---

## Flujo de trabajo con Git

Este proyecto trabaja sobre **una sola rama: `main`**.

```bash
# Antes de pushear, verificar que el servidor arranca sin errores
uvicorn main:app --host 0.0.0.0 --port 8001

# Probar los endpoints con Swagger
# http://localhost:8001/docs
```

---

## Convención de commits

Formato: `<tipo>(<ámbito>): <descripción corta>`

| Tipo | Cuándo usarlo |
|------|---------------|
| `feat` | Nuevo endpoint o funcionalidad de IA |
| `fix` | Corrección de bug en prompt, lógica o modelo |
| `model` | Cambios en datos sintéticos, hiperparámetros o features del ML |
| `prompt` | Cambios solo en SYSTEM_PROMPT de Groq |
| `refactor` | Limpieza sin nueva funcionalidad |
| `docs` | Solo documentación |
| `chore` | Dependencias, variables de entorno |

### Ámbitos
```
diagrama, formulario, analisis, main, schemas, deploy
```

### Ejemplos
```
feat(formulario): agregar tipo de campo FIRMA_DIGITAL
fix(diagrama): corregir validación de nodos PARALELO sin JOIN
model(analisis): ajustar distribuciones para mejor solapamiento entre clases
prompt(diagrama): agregar ejemplo de FORK/JOIN al SYSTEM_PROMPT
chore(deps): actualizar groq-sdk a 0.8.0
```

---

## Qué NO subir a git

```gitignore
.env                          # Clave de Groq
venv/                         # Entorno virtual
modelo_cuello_botella.pkl     # Modelo entrenado (85 MB)
__pycache__/
*.pyc
```

> El `.pkl` puede subirse explícitamente si se quiere fijar una versión específica del modelo para producción. En ese caso usar `git lfs` por el tamaño.

---

## Checklist antes de commitear

- [ ] `uvicorn main:app` arranca sin errores
- [ ] `GET /health` responde `{"status": "ok"}`
- [ ] Si se cambió el modelo: el `.pkl` nuevo funciona correctamente
- [ ] Si se cambió el SYSTEM_PROMPT: probado con al menos 3 prompts distintos
- [ ] No se subió el `.env` ni el `venv/`

---

## Compatibilidad con el backend Java

El backend Java llama a este servicio vía HTTP en `IaService.java`. Los contratos de los endpoints (campos del JSON) deben mantenerse estables. Si se cambia un campo de respuesta, también hay que actualizar `IaController.java` y los DTOs en el backend.

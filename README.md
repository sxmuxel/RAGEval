# Evaluación Pipeline RAG

## 1. Introducción

Este proyecto evalúa un pipeline de Retrieval-Augmented Generation (RAG) construido sobre artículos del reglamento académico institucional.

### Métricas evaluadas
- **Faithfulness:** fidelidad de la respuesta respecto al contexto recuperado.
- **Answer Relevancy:** pertinencia de la respuesta frente a la pregunta.
- **Context Precision:** precisión de los chunks recuperados.

---

## 2. Parámetros del Pipeline

| Parámetro | Valor |
|---|---|
| Documentos | Artículos del Reglamento Estudiantil |
| Embeddings | TF-IDF (local) |
| chunk_size / overlap | 1 artículo / 0 |
| k (chunks recuperados) | 3 |
| LLM Generador | llama-3.3-70b-versatile |
| LLM Juez (RAGAS) | llama-3.3-70b-versatile |

---

## 3. Casos de Prueba

Se utilizaron 8 preguntas distribuidas en 4 tipos de evaluación.

| Tipo | Descripción | Qué evalúa |
|---|---|---|
| A – Textual | Respuesta literal en el documento | Recuperación exacta y generación fiel |
| B – Vocabulario diferente | Uso de sinónimos o paráfrasis | Calidad semántica de embeddings |
| C – Multi-chunk | Requiere combinar varios artículos | Síntesis sobre múltiples contextos |
| D – Fuera de alcance | El documento no contiene la respuesta | Detección de alucinaciones |

## 4. Instrucciones de ejecución (Windows)

### Paso 1: Clonar repositorio

En la terminal:
```bash
git clone https://github.com/sxmuxel/RAGEval.git
cd RAGEval
```

### Paso 2: Crear entorno virtual

En la terminal:
```bash
python -m venv venv
.\venv\Scripts\activate
```

### Paso 3: Instalar las dependencias

En la terminal:
```bash
pip install -r requirements.txt
```

### Paso 4: Ejecutar el script

En la terminal:
```bash
python main.py
```

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import Dataset
from openai import OpenAI                      # Groq usa el cliente OpenAI
from langchain_openai import ChatOpenAI        # igual para RAGAS
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY no encontrada en .env")

GROQ_BASE_URL   = "https://api.groq.com/openai/v1"
GENERATOR_MODEL = "llama-3.3-70b-versatile"
JUDGE_MODEL     = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "TF-IDF (sklearn, local)"
CHUNK_SIZE      = 1
CHUNK_OVERLAP   = 0
K_RETRIEVED     = 3

documentos_reglamento = [
    "Artículo 31: La matrícula otorga el derecho a cursar el programa académico y debe renovarse dentro de los plazos del calendario académico.",
    "Artículo 42: La asistencia de los estudiantes es obligatoria en espacios formativos con presencialidad o mediación tecnológica sincrónica.",
    "Artículo 43: El profesor debe llevar control estricto de la asistencia de los estudiantes a clases y eventos académicos obligatorios.",
    "Artículo 49: El estudiante debe realizar el registro oportuno de actividades académicas y sus modificaciones cuando sea necesario.",
    "Artículo 51: El estudiante puede registrar asignaturas de diversos periodos con dispersión máxima de cuatro periodos según los prerrequisitos.",
    "Artículo 52: La cancelación total de asignaturas equivale a retiro voluntario del periodo académico.",
    "Artículo 89: La Institución otorga distinciones de Honor y ofrece incentivos como monitorías e investigación científica a estudiantes destacados.",
    "Artículo 90: La Mención de Honor reconoce estudiantes con promedio mínimo de cuarenta y tres sobre cincuenta y sin sanciones disciplinarias.",
    "Artículo 93: El Consejo Académico concede Grado de Honor o Grado Meritorio a estudiantes con los mejores promedios del programa.",
    "Artículo 97: Las monitorías vinculan estudiantes sobresalientes a actividades de docencia, investigación y proyección social institucional.",
]

test_cases = [
    {"tipo": "T1 - Literal",
     "question": "¿Qué otorga la matrícula al estudiante?",
     "ground_truth": "La matrícula otorga el derecho a cursar el programa académico y debe renovarse dentro de los plazos del calendario académico."},
    {"tipo": "T1 - Literal",
     "question": "¿Qué equivale la cancelación total de asignaturas?",
     "ground_truth": "La cancelación total de asignaturas equivale a retiro voluntario del periodo académico."},
    {"tipo": "T2 - Vocabulario dif.",
     "question": "¿Es obligatorio asistir a las clases presenciales o virtuales en tiempo real?",
     "ground_truth": "Sí, la asistencia es obligatoria en espacios formativos con presencialidad o mediación tecnológica sincrónica."},
    {"tipo": "T2 - Vocabulario dif.",
     "question": "¿Puede un alumno brillante participar en tareas de enseñanza o investigación?",
     "ground_truth": "Sí, las monitorías vinculan estudiantes sobresalientes a actividades de docencia, investigación y proyección social institucional."},
    {"tipo": "T3 - Multi-chunk",
     "question": "¿Qué reconocimientos puede obtener un estudiante con excelente desempeño académico?",
     "ground_truth": "Un estudiante destacado puede obtener distinción de Honor (Art. 89), Mención de Honor si tiene promedio mínimo 43/50 sin sanciones (Art. 90), Grado de Honor o Meritorio del Consejo Académico (Art. 93), y participar en monitorías de docencia e investigación (Art. 97)."},
    {"tipo": "T3 - Multi-chunk",
     "question": "¿Cuáles son las obligaciones del estudiante respecto a registro y asistencia?",
     "ground_truth": "El estudiante debe asistir obligatoriamente a espacios formativos (Art. 42), realizar registro oportuno de actividades (Art. 49), y puede registrar asignaturas de hasta cuatro periodos distintos según prerrequisitos (Art. 51)."},
    {"tipo": "T4 - Fuera alcance",
     "question": "¿Cuál es el monto de la matrícula semestral para estudiantes de posgrado?",
     "ground_truth": "El reglamento no contiene información sobre montos de matrícula para posgrado."},
    {"tipo": "T4 - Fuera alcance",
     "question": "¿Cuántos créditos tiene cada asignatura del programa de Ingeniería de Sistemas?",
     "ground_truth": "El reglamento no especifica el número de créditos por asignatura ni menciona programas específicos."},
]

print("Construyendo índice TF-IDF...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
doc_matrix = vectorizer.fit_transform(documentos_reglamento)
print(f"Índice listo: {doc_matrix.shape[0]} docs × {doc_matrix.shape[1]} features")

def retrieve(question, k=K_RETRIEVED):
    q_vec = vectorizer.transform([question])
    sims  = cosine_similarity(q_vec, doc_matrix).flatten()
    top_k = np.argsort(sims)[::-1][:k]
    return [documentos_reglamento[i] for i in top_k]

groq_client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

def generate_answer(question, contexts):
    ctx_text = "\n".join(f"- {c}" for c in contexts)
    resp = groq_client.chat.completions.create(
        model=GENERATOR_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": (
                "Eres un asistente universitario. Responde ÚNICAMENTE con base en "
                "el contexto del reglamento estudiantil proporcionado. "
                "Si la información no está en el contexto, responde exactamente: "
                '"El reglamento no contiene información sobre este tema."'
            )},
            {"role": "user", "content": f"Contexto:\n{ctx_text}\n\nPregunta: {question}"}
        ]
    )
    return resp.choices[0].message.content.strip()

print("\nGenerando respuestas con Groq...")
questions, answers, contexts_list, ground_truths, tipos = [], [], [], [], []

for i, case in enumerate(test_cases):
    q   = case["question"]
    ctx = retrieve(q)
    ans = generate_answer(q, ctx)
    questions.append(q)
    answers.append(ans)
    contexts_list.append(ctx)
    ground_truths.append(case["ground_truth"])
    tipos.append(case["tipo"])
    print(f"  [{i+1}/{len(test_cases)}] {case['tipo']} → OK")

ragas_dataset = Dataset.from_dict({
    "question":     questions,
    "answer":       answers,
    "contexts":     contexts_list,
    "ground_truth": ground_truths,
})

print("\nEvaluando con RAGAS (juez: Groq/Llama)...")

judge_llm = ChatOpenAI(
    model=JUDGE_MODEL,
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL,
    temperature=0,
)

result = evaluate(
    dataset=ragas_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=judge_llm,
    raise_exceptions=False,
)

SEP = "=" * 70
print(f"\n{SEP}\nRESULTADOS DE EVALUACIÓN RAG\n{SEP}")

print("\nPARÁMETROS DEL PIPELINE:")
for k, v in {
    "Documentos":              f"{len(documentos_reglamento)} artículos del reglamento",
    "Modelo de embeddings":    EMBEDDING_MODEL,
    "chunk_size / overlap":    f"{CHUNK_SIZE} artículo / {CHUNK_OVERLAP}",
    "k (chunks recuperados)":  K_RETRIEVED,
    "LLM generador":           GENERATOR_MODEL,
    "LLM juez (RAGAS)":        JUDGE_MODEL,
}.items():
    print(f"  {k:<30} {v}")

result_df = result.to_pandas()
result_df["tipo"]     = tipos
result_df["question"] = questions
result_df["answer"]   = answers

print("\nMÉTRICAS POR PREGUNTA:")
print(f"{'#':<3} {'Tipo':<22} {'Faithfulness':>13} {'Answer Rel.':>12} {'Ctx Prec.':>10}")
print("-" * 65)
for i, row in result_df.iterrows():
    f  = row.get("faithfulness",      float("nan"))
    ar = row.get("answer_relevancy",  float("nan"))
    cp = row.get("context_precision", float("nan"))
    print(f"{i+1:<3} {tipos[i]:<22} {f:>13.3f} {ar:>12.3f} {cp:>10.3f}")

f_mean  = result_df["faithfulness"].mean()
ar_mean = result_df["answer_relevancy"].mean()
cp_mean = result_df["context_precision"].mean()
print("-" * 65)
print(f"{'PROMEDIO':<25} {f_mean:>13.3f} {ar_mean:>12.3f} {cp_mean:>10.3f}")

print("\nDETALLE DE RESPUESTAS:")
for i, (q, a, ctx) in enumerate(zip(questions, answers, contexts_list)):
    print(f"\n[{i+1}] {tipos[i]}")
    print(f"  P: {q}")
    print(f"  R: {a[:300]}{'...' if len(a) > 300 else ''}")
    print(f"  Chunks ({len(ctx)}):")
    for c in ctx:
        print(f"    · {c[:85]}...")

result_df.to_csv("resultados_ragas.csv", index=False)
print("\n✅ Evaluación completada. Resultados en resultados_ragas.csv")
import os
from dotenv import load_dotenv
from google import genai
from groq import Groq

load_dotenv()
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Generation model IDs are configurable. The defaults are the original IDs (kept for API/test
# compatibility) but both have been discontinued upstream; see README "Model configuration".
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def gemini_model() -> str:
    return os.environ.get("DOCINTEL_GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def groq_model() -> str:
    return os.environ.get("DOCINTEL_GROQ_MODEL", DEFAULT_GROQ_MODEL)


def groq_label() -> str:
    return "groq-llama-3.3" if groq_model() == DEFAULT_GROQ_MODEL else f"groq:{groq_model()}"

SYSTEM_PROMPTS = {
    "enterprise": """You are an enterprise knowledge assistant. Answer questions using ONLY the provided company documents (SOPs, policies, manuals). Be precise and professional. Always cite the source document.""",
    "legal": """You are a legal document analyst. Simplify complex legal language into plain English. Identify risky clauses, unusual terms, and summarize what the user is agreeing to. Always cite the source document. Flag anything that seems unfavorable with ⚠️.""",
    "medical": """You are a medical report explainer. Explain medical terms, test results, and diagnoses in simple, clear language a patient can understand. Suggest questions the user could ask their doctor. Always cite the source document. Never diagnose — always recommend consulting a doctor.""",
    "study": """You are a study assistant. Answer questions, generate summaries, create flashcard-style Q&A pairs, and explain concepts from the uploaded study material. Be clear and educational. Always cite the source document."""
}

def build_prompt(chunks: list[dict], question: str) -> str:
    context = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)
    return f"Context:\n{context}\n\nQuestion: {question}"

def get_confidence(chunks: list[dict]) -> str:
    if not chunks:
        return "low"
    avg_len = sum(len(c["text"]) for c in chunks) / len(chunks)
    if avg_len > 300:
        return "high"
    elif avg_len > 150:
        return "medium"
    return "low"

def log_query(question: str, answer: str, mode: str, sources: list):
    import csv, datetime
    os.makedirs("data", exist_ok=True)
    with open("data/audit_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.datetime.now().isoformat(),
            mode, question,
            answer[:100] + "...",
            ", ".join(sources)
        ])

def ask_gemini(chunks: list[dict], question: str, mode: str) -> str:
    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["enterprise"])
    response = gemini_client.models.generate_content(
        model=gemini_model(),
        contents=f"{system_prompt}\n\n{build_prompt(chunks, question)}"
    )
    return response.text

def ask_groq(chunks: list[dict], question: str, mode: str) -> str:
    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["enterprise"])
    response = groq_client.chat.completions.create(
        model=groq_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_prompt(chunks, question)}
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content

def ask(chunks: list[dict], question: str, mode: str = "enterprise") -> dict:
    try:
        answer = ask_gemini(chunks, question, mode)
        model_used = gemini_model()
    except Exception as e:
        print(f"Gemini failed ({e}), falling back to Groq...")
        answer = ask_groq(chunks, question, mode)
        model_used = groq_label()
    confidence = get_confidence(chunks)
    sources = list(set(c["source"] for c in chunks))
    log_query(question, answer, mode, sources)
    return {"answer": answer, "model": model_used, "confidence": confidence}
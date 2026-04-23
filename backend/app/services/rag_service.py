from transformers import pipeline
from backend.app.services.vector_service import search_similar_records

generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)


def build_context(matches):
    context_blocks = []

    for i, match in enumerate(matches, start=1):
        context_blocks.append(
            f"Record {i}:\n"
            f"Clinical Note: {match.get('clinical_note', '')}\n"
            f"Diagnoses: {match.get('diagnoses', '')}\n"
            f"Symptoms: {match.get('symptoms', '')}\n"
            f"Medications: {match.get('medications', '')}\n"
            f"Distance: {match.get('distance', '')}\n"
        )

    return "\n".join(context_blocks)


def strict_clean_answer(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    wanted_prefixes = [
        "Summary:",
        "Likely diagnosis:",
        "Key symptoms:",
        "Relevant medications:"
    ]

    cleaned_lines = []
    seen = set()

    for line in lines:
        for prefix in wanted_prefixes:
            if line.lower().startswith(prefix.lower()) and prefix not in seen:
                cleaned_lines.append(line)
                seen.add(prefix)
                break

    return "\n".join(cleaned_lines).strip()


def fallback_summary_from_matches(matches: list[dict]) -> str:
    top = matches[0]

    diagnosis = top.get("diagnoses", "not identified")
    symptoms = top.get("symptoms", "not identified")
    medications = top.get("medications", "not identified")

    return (
        f"Summary: The query is most similar to a previously stored clinical case.\n"
        f"Likely diagnosis: {diagnosis}\n"
        f"Key symptoms: {symptoms}\n"
        f"Relevant medications: {medications}"
    )


def generate_rag_response(query: str, top_k: int = 3):
    retrieval_results = search_similar_records(query, top_k=top_k)
    matches = retrieval_results.get("matches", [])

    if not matches:
        return {
            "query": query,
            "matches": [],
            "generated_answer": "No similar clinical records were found."
        }

    context = build_context(matches)

    prompt = f"""<|system|>
You are a clinical reasoning assistant.
Use only the retrieved records below.
Synthesize the most relevant information from the top matches.
Return exactly 4 lines in this format:

Summary: ...
Likely diagnosis: ...
Key symptoms: ...
Relevant medications: ...

Do not invent facts. Use only the retrieved records.
<|user|>
Query: {query}

Retrieved records:
{context}
<|assistant|>
"""

    try:
        output = generator(
            prompt,
            max_new_tokens=120,
            do_sample=False,
            truncation=True
        )[0]["generated_text"]

        generated_answer = output.split("<|assistant|>", 1)[-1].strip()
        generated_answer = strict_clean_answer(generated_answer)

        if not generated_answer or "Likely diagnosis:" not in generated_answer:
            raise ValueError("Weak generation")

    except Exception:
        generated_answer = fallback_summary_from_matches(matches)

    return {
        "query": query,
        "matches": matches,
        "generated_answer": generated_answer
    }
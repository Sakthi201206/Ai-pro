import re

from backend.core.prompt_builder import build_prompt
from backend.services.llm_service import generate_response


def generate_debate(topic: str) -> dict:
    """
    Full RAG pipeline:
      1. Retrieve relevant docs via FAISS
      2. Build a structured prompt
      3. Generate debate response (LLM or template)
      4. Return structured JSON
    """
    from backend.core.retriever import retrieve

    docs = retrieve(topic, k=20)

    prompt = build_prompt(topic, docs)
    response_text = generate_response(prompt, topic, docs)

    # Parse the structured response into sections
    sections = _parse_response(response_text)

    if not sections.get("FOR") or not sections.get("AGAINST"):
        response_text = _template_generate(topic, docs)
        sections = _parse_response(response_text)

    return {
        "topic":    topic,
        "result":   response_text,
        "for":      sections.get("FOR", []),
        "against":  sections.get("AGAINST", []),
        "verdict":  sections.get("VERDICT", ""),
        "sources":  [
            {
                "text":     d["text"],
                "label":    d["label"],
                "source":   d["source"],
                "year":     d["year"],
                "strength": d["strength"],
                "score":    round(d["score"], 4),
            }
            for d in docs[:10]
        ],
    }

def _parse_response(text: str) -> dict:
    """Parse ### FOR / ### AGAINST / ### VERDICT sections."""
    sections = {}
    current = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        normalized = line.upper()
        if normalized.startswith("### FOR") or normalized.startswith("FOR:"):
            current = "FOR"
            sections[current] = []
            continue
        if normalized.startswith("### AGAINST") or normalized.startswith("AGAINST:"):
            current = "AGAINST"
            sections[current] = []
            continue
        if normalized.startswith("### VERDICT") or normalized.startswith("VERDICT:"):
            current = "VERDICT"
            sections[current] = ""
            continue

        if current in ("FOR", "AGAINST"):
            item_match = re.match(r'^[\d]+[\.)]\s*(.*)', line)
            if item_match:
                sections[current].append(item_match.group(1).strip())
                continue
            if line.startswith("- ") or line.startswith("* "):
                sections[current].append(line[2:].strip())
                continue

        if current == "VERDICT":
            sections[current] += (" " if sections[current] else "") + line

    return sections

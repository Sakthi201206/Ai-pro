def build_prompt(topic: str, context_docs: list[dict]) -> str:
    """
    Build a structured prompt for debate generation.
    context_docs is a list of retrieved document dicts with keys:
      text, label, source, year, strength
    """
    pro_docs  = [d for d in context_docs if d["label"] == "pro"]
    con_docs  = [d for d in context_docs if d["label"] == "con"]
    neu_docs  = [d for d in context_docs if d["label"] == "neutral"]

    def fmt(docs):
        return "\n".join(
            f"  • [{d['source'].upper()} {d['year']}] {d['text']} (strength: {d['strength']:.2f})"
            for d in docs[:5]
        ) or "  (none retrieved)"

    prompt = f"""You are an expert AI Debate Analyst. Given a debate topic and retrieved evidence from research papers, news articles, and social media, generate a well-structured debate response.

TOPIC: "{topic}"

RETRIEVED EVIDENCE:
PRO (supporting) arguments:
{fmt(pro_docs)}

CON (opposing) arguments:
{fmt(con_docs)}

NEUTRAL / CONTEXTUAL observations:
{fmt(neu_docs)}

INSTRUCTIONS:
1. Write 4–5 strong, distinct arguments FOR the topic. Each argument should be 1–2 sentences. Base them on the evidence above, but expand and elaborate creatively to make them more comprehensive and convincing.
2. Write 4–5 strong, distinct arguments AGAINST the topic. Each argument should be 1–2 sentences. Base them on the evidence above, but expand and elaborate creatively to make them more comprehensive and convincing.
3. Write a brief 2–3 sentence balanced VERDICT summarising both sides.

FORMAT your response exactly as shown below. If there is no retrieved evidence, still write balanced arguments based on the topic itself:

### FOR
1. ...
2. ...
3. ...
4. ...
5. ...

### AGAINST
1. ...
2. ...
3. ...
4. ...
5. ...

### VERDICT
...
"""
    return prompt

from backend.config import OPENAI_API_KEY


def generate_response(prompt: str, topic: str, context_docs: list[dict]) -> str:
    """
    Generate a debate response.
    - If OPENAI_API_KEY is set → call GPT-4o-mini
    - If the OpenAI request fails, fall back to template generation
    """
    if OPENAI_API_KEY:
        response = _openai_generate(prompt)
        if response is not None:
            return response
        print("OpenAI failed or quota exhausted. Falling back to template generation.")
    return _template_generate(topic, context_docs)


# ── OpenAI path ──────────────────────────────────────────────
def _openai_generate(prompt: str) -> str | None:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=900,
        )
        return response.choices[0].message.content
    except Exception as exc:
        print("OpenAI API error or unexpected failure:", repr(exc))
        return None


# ── Template-based fallback (no API key needed) ───────────────
def _template_generate(topic: str, docs: list[dict]) -> str:
    """Build a structured debate response purely from retrieved docs."""

    def dedupe(items):
        seen, out = set(), []
        for x in items:
            if x["text"] not in seen:
                seen.add(x["text"])
                out.append(x)
        return out

    pros = dedupe(sorted(
        [d for d in docs if d["label"] == "pro"],
        key=lambda d: d["strength"] * d["score"], reverse=True
    ))[:5]

    cons = dedupe(sorted(
        [d for d in docs if d["label"] == "con"],
        key=lambda d: d["strength"] * d["score"], reverse=True
    ))[:5]

    neutrals = dedupe([d for d in docs if d["label"] == "neutral"])[:3]

    def arg_line(i, d):
        # Create more elaborate arguments based on the evidence
        base_text = d['text']
        src = f"[{d['source'].title()}, {d['year']}]"

        # Expand the argument based on the label and content
        if d['label'] == 'pro':
            if 'create' in base_text.lower() or 'opportunity' in base_text.lower():
                expanded = f"AI will create new job opportunities and transform existing roles, leading to economic growth and innovation. {base_text}."
            elif 'reduce' in base_text.lower() or 'error' in base_text.lower():
                expanded = f"AI improves workplace efficiency and safety by reducing human error and automating dangerous tasks. {base_text}."
            elif 'technology' in base_text.lower() and 'create' in base_text.lower():
                expanded = f"Throughout history, technological advancements have consistently created more jobs than they eliminate. {base_text}."
            else:
                expanded = f"AI brings significant benefits to the workforce and economy. {base_text}."
        elif d['label'] == 'con':
            if 'replace' in base_text.lower() or 'automation' in base_text.lower():
                expanded = f"AI and automation threaten to displace workers, particularly in low-skilled positions, leading to unemployment. {base_text}."
            elif 'risk' in base_text.lower() or 'at risk' in base_text.lower():
                expanded = f"Certain worker groups face heightened job displacement risks from AI implementation. {base_text}."
            elif 'unemployment' in base_text.lower():
                expanded = f"AI adoption could result in widespread unemployment and economic disruption. {base_text}."
            elif 'inequality' in base_text.lower():
                expanded = f"AI may exacerbate economic and social inequalities in society. {base_text}."
            else:
                expanded = f"AI poses significant challenges and risks to employment. {base_text}."
        else:  # neutral
            expanded = f"The impact of AI on jobs remains uncertain and requires careful consideration. {base_text}."

        return f"{i}. {expanded} {src} (argument strength: {d['strength']:.2f})"

    pro_block = "\n".join(arg_line(i+1, d) for i, d in enumerate(pros))
    con_block = "\n".join(arg_line(i+1, d) for i, d in enumerate(cons))

    if not pro_block:
        generic_pros = _generate_generic_arguments(topic, "pro", 5)
        pro_block = "\n".join(f"{i+1}. {arg}" for i, arg in enumerate(generic_pros))
    if not con_block:
        generic_cons = _generate_generic_arguments(topic, "con", 5)
        con_block = "\n".join(f"{i+1}. {arg}" for i, arg in enumerate(generic_cons))

    # Build verdict from neutral docs and overall balance
    if neutrals:
        verdict_detail = " ".join(d["text"] + "." for d in neutrals[:2])
    else:
        verdict_detail = f"{topic} raises multiple considerations and should be evaluated carefully."

    pro_count = len(pros) if pros else 0
    con_count = len(cons) if cons else 0
    total_evidence = pro_count + con_count

    if total_evidence == 0:
        leaning = "No clear evidence available from the retrieval dataset"
    elif pro_count > con_count:
        leaning = f"Evidence leans slightly PRO ({pro_count} vs {con_count} arguments)"
    elif con_count > pro_count:
        leaning = f"Evidence leans slightly CON ({con_count} vs {pro_count} arguments)"
    else:
        leaning = f"Evidence is evenly balanced ({pro_count} PRO and {con_count} CON arguments)"

    verdict = (
        f"{verdict_detail} "
        f"{leaning}. "
        f"Good debate preparation should use both topic context and any available evidence."
    )

    return f"""### FOR
{pro_block}

### AGAINST
{con_block}

### VERDICT
{verdict}"""


def _generate_generic_arguments(topic: str, label: str, count: int = 5) -> list[str]:
    topic_text = topic.strip().rstrip('.')
    if label == "pro":
        templates = [
            f"{topic_text} highlights positive competition and gives people more choice.",
            f"{topic_text} encourages innovation and better products by driving brands or ideas to improve.",
            f"{topic_text} can help clarify what matters most for consumers and stakeholders.",
            f"{topic_text} supports a stronger market dynamic that often benefits quality and variety.",
            f"{topic_text} is useful for generating new perspectives and healthy debate."
        ]
    else:
        templates = [
            f"{topic_text} can distract from more important issues and create unnecessary rivalry.",
            f"{topic_text} risks promoting superficial comparisons instead of real value.",
            f"{topic_text} can increase pressure on participants and reduce focus on long-term sustainability.",
            f"{topic_text} may lead to polarized opinions rather than thoughtful consensus.",
            f"{topic_text} might exaggerate differences while overlooking shared benefits."
        ]
    return templates[:count]

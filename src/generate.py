"""
Handles answer generation via Groq's free-tier API.

>>> Requires GROQ_API_KEY to be set (see .env.example) <<<
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import GROQ_API_KEY, GROQ_MODEL, PROMPT_TEMPLATES

_client = None


def get_groq_client():
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Create a free key at "
                "https://console.groq.com and put it in a .env file "
                "(see .env.example) as GROQ_API_KEY=your_key_here"
            )
        from groq import Groq
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def generate_answer(question: str, context_chunks: list, prompt_style: str = "loose") -> str:
    """
    context_chunks: list of chunk text strings (already retrieved/reranked).
    prompt_style: "strict" or "loose" — see config.PROMPT_TEMPLATES.
    """
    client = get_groq_client()
    context_text = "\n\n---\n\n".join(context_chunks)
    prompt = PROMPT_TEMPLATES[prompt_style].format(context=context_text, question=question)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # deterministic-ish answers make evaluation runs more comparable
    )
    return response.choices[0].message.content

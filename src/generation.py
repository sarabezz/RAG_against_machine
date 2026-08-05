from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"

SYSTEM_PROMPT = (
    "Eres un asistente que responde preguntas sobre el codigo y la "
    "documentacion del proyecto vLLM. Usa unicamente la informacion "
    "del contexto proporcionado. Si el contexto no contiene la "
    "respuesta, dilo claramente. Responde de forma breve y precisa."
)


@lru_cache(maxsize=1)
def load_model():
    """Carga el modelo y el tokenizer una sola vez por proceso."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto", device_map="auto"
    )
    return tokenizer, model


def read_source_text(source: Dict[str, Any]) -> str:
    """Recupera el texto real de una fuente a partir de su ubicacion."""
    path = Path(source["file_path"])
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[source["first_character_index"]:source["last_character_index"]]


def build_context(sources: List[Dict[str, Any]],
                   max_chars: int = 6000) -> str:
    """Concatena el texto de las fuentes respetando un limite de caracteres."""
    partes = []
    total = 0
    for src in sources:
        texto = read_source_text(src)
        fragmento = f"### {src['file_path']}\n{texto}\n"
        if total + len(fragmento) > max_chars:
            break
        partes.append(fragmento)
        total += len(fragmento)
    return "\n".join(partes)


def generate_answer(question: str, sources: List[Dict[str, Any]],
                     max_new_tokens: int = 150) -> str:
    tokenizer, model = load_model()
    context = build_context(sources)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Contexto:\n{context}\n\nPregunta: {question}"},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        do_sample=True,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
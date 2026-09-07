import math
import random
import time

import requests

from common import console, settings


class TruncatedAnswer(RuntimeError):
    pass


CONTEXT_BLOCK = 512

active_model = None

chars_per_token = None

context_in_use = 0

model_cards = {}

operations_done = 0


def model_name():
    return settings.current().model or settings.OLLAMA_MODEL


def model_card(name):
    if name not in model_cards:
        try:
            answer = requests.post(
                f"{settings.OLLAMA_HOST}/api/show", json={"model": name}, timeout=30
            )
            answer.raise_for_status()
            model_cards[name] = answer.json() or {}
        except Exception as error:
            console.warn(f"Could not read the Ollama card of '{name}': {error}")
            model_cards[name] = {}
    return model_cards[name]


def model_thinks(name):
    return "thinking" in (model_card(name).get("capabilities") or [])


def model_context_limit(name):
    information = model_card(name).get("model_info") or {}
    lengths = [
        value
        for key, value in information.items()
        if key.endswith(".context_length") and isinstance(value, int) and value > 0
    ]
    return max(lengths) if lengths else None


def context_ceiling(name):
    if settings.CONTEXT_LIMIT > 0:
        return settings.CONTEXT_LIMIT

    limit = model_context_limit(name)
    if limit:
        return limit

    console.once(
        f"context-fallback-{name}",
        f"Ollama did not report the context limit of '{name}'; capping the window at "
        f"{settings.CONTEXT_LIMIT_FALLBACK} tokens (OLLAMA_CTX_MAX_FALLBACK).",
        console.warn,
    )
    return settings.CONTEXT_LIMIT_FALLBACK


def estimate_tokens(prompt):
    ratio = chars_per_token or settings.CHARS_PER_TOKEN
    return int(math.ceil(len(prompt or "") / ratio))


def calibrate(prompt, prompt_tokens):
    global chars_per_token
    if not prompt or not prompt_tokens or prompt_tokens <= 0:
        return

    measured = min(max(len(prompt) / prompt_tokens, 1.5), 8.0)
    if chars_per_token is None or measured < chars_per_token:
        chars_per_token = measured


def window_for(prompt, margin, ceiling):
    global context_in_use
    if settings.CONTEXT_SIZE > 0:
        return settings.CONTEXT_SIZE

    estimated = estimate_tokens(prompt)
    if margin <= 0:
        needed = ceiling
    else:
        needed = int(math.ceil((estimated + margin) / CONTEXT_BLOCK)) * CONTEXT_BLOCK
        needed = min(needed, ceiling)

    if needed > context_in_use:
        console.detail(
            f"Context window set to {needed} tokens (prompt estimated at {estimated}, "
            f"{needed - estimated} left for the answer)."
        )
        context_in_use = needed
    return context_in_use


def current_window():
    if settings.CONTEXT_SIZE > 0:
        return settings.CONTEXT_SIZE
    return context_in_use or None


def describe_window(name):
    if settings.CONTEXT_SIZE > 0:
        return f"fixed at {settings.CONTEXT_SIZE} tokens"
    ceiling = context_ceiling(name)
    if settings.CONTEXT_MARGIN > 0:
        return (f"automatic — prompt + {settings.CONTEXT_MARGIN} tokens of headroom, growing to "
                f"{ceiling} when an answer needs it")
    return f"automatic, already at the model maximum ({ceiling})"


def announce(name=None):
    global active_model
    name = name or model_name()
    if active_model != name:
        console.info(f"Using '{name}' through Ollama at {settings.OLLAMA_HOST}.")
        console.detail(f"Context window: {describe_window(name)}.")
        console.detail(
            "Reasoning: this model does not reason; nothing to switch off."
            if not model_thinks(name)
            else "Reasoning: off — the reading pass does that work once per snippet."
        )
        active_model = name
    return name


def unload():
    global active_model
    if active_model is None:
        return

    name = active_model
    try:
        requests.post(
            f"{settings.OLLAMA_HOST}/api/chat",
            json={"model": name, "messages": [], "keep_alive": 0},
            timeout=None,
        )
        console.detail(f"Model '{name}' unloaded from Ollama.")
    except Exception as error:
        console.warn(f"Could not ask Ollama to unload '{name}': {error}")
    finally:
        active_model = None


def load_into_memory(name):
    payload = {"model": name, "messages": [], "keep_alive": settings.OLLAMA_KEEP_ALIVE}
    window = current_window()
    if window:
        payload["options"] = {"num_ctx": window}

    try:
        answer = requests.post(f"{settings.OLLAMA_HOST}/api/chat", json=payload, timeout=None)
        answer.raise_for_status()
        console.detail(f"Model '{name}' reloaded in Ollama.")
    except Exception as error:
        console.warn(f"Could not reload '{name}' ({error}); the next request will load it.")


def recycle(reason=""):
    name = active_model or model_name()
    console.detail(f"Recycling the model ({reason})…")

    try:
        unload()
    except Exception as error:
        console.warn(f"Could not unload the model for recycling: {error}")

    from common import cleanup
    cleanup.empty_gpu_cache()

    if settings.RECYCLE_PAUSE > 0:
        time.sleep(settings.RECYCLE_PAUSE)

    announce(name)
    load_into_memory(name)


def count_operation(kind="request"):
    global operations_done
    if settings.RECYCLE_EVERY <= 0:
        return

    operations_done += 1
    if operations_done < settings.RECYCLE_EVERY:
        return

    operations_done = 0
    try:
        recycle(f"{settings.RECYCLE_EVERY} operations done, the last a {kind}")
    except Exception as error:
        console.warn(f"Model recycling failed ({error}); continuing without a reload.")


def chat(prompt, window, max_tokens=None, name=None):
    payload = {
        "model": name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "keep_alive": settings.OLLAMA_KEEP_ALIVE,
        "options": {
            "num_ctx": window,
            "num_predict": max_tokens or settings.MAX_ANSWER_TOKENS,
            "temperature": 0.0,
            "seed": settings.SEED,
        },
    }

    if model_thinks(name):
        payload["think"] = False

    answer = requests.post(f"{settings.OLLAMA_HOST}/api/chat", json=payload, timeout=None)
    answer.raise_for_status()
    reply = answer.json()

    calibrate(prompt, reply.get("prompt_eval_count"))

    text = (reply.get("message", {}).get("content") or "").strip()
    return text, reply


def truncation_reason(reply, window, max_tokens):
    prompt_tokens = reply.get("prompt_eval_count") or 0
    generated = reply.get("eval_count") or 0

    if max_tokens > 0 and generated >= max_tokens:
        return (f"generation hit the OLLAMA_NUM_PREDICT ceiling ({max_tokens}). Use "
                "OLLAMA_NUM_PREDICT=-1 to let the model decide when to stop")

    if settings.CONTEXT_SIZE > 0:
        return (f"the context ran out — prompt ({prompt_tokens}) + generation ({generated}) filled "
                f"the fixed window of {window}. Raise OLLAMA_NUM_CTX, or leave it empty so the "
                "window is sized and grown automatically")

    source = "pinned in OLLAMA_CTX_MAX" if settings.CONTEXT_LIMIT > 0 else "of the model itself"
    return (f"the context ran out — prompt ({prompt_tokens}) + generation ({generated}) filled the "
            f"{window}-token window, already the ceiling {source}. Use a model with a larger "
            "context or shorten the prompt")


def ask(prompt, max_tokens=None):
    name = announce()
    random.seed(settings.SEED)

    ceiling = context_ceiling(name)
    margin = settings.CONTEXT_MARGIN

    while True:
        window = window_for(prompt, margin, ceiling)
        text, reply = chat(prompt, window, max_tokens=max_tokens, name=name)
        if text:
            return text

        thinking = (reply.get("message", {}).get("thinking") or "").strip()
        if not thinking and reply.get("done_reason") != "length":
            return text

        limit = max_tokens or settings.MAX_ANSWER_TOKENS
        generated = reply.get("eval_count") or 0
        hit_token_ceiling = limit > 0 and generated >= limit

        if not hit_token_ceiling and settings.CONTEXT_SIZE <= 0 and window < ceiling:
            margin = margin * 2 if margin > 0 else 0
            grown = window_for(prompt, margin, ceiling)
            if grown > window:
                console.detail(
                    f"The answer filled the {window}-token window ({generated} tokens); "
                    f"reissuing with {grown}."
                )
                continue

        raise TruncatedAnswer(
            f"'{name}' finished with no answer (done_reason={reply.get('done_reason')}, "
            f"{generated} tokens generated, {len(thinking)} characters of reasoning): "
            f"{truncation_reason(reply, window, limit)}."
        )


def parameter_billions(reported):
    text = (reported or "").strip().upper().rstrip("B")
    try:
        return float(text)
    except ValueError:
        return None


def is_heavy(billions):
    if settings.OLLAMA_MAX_PARAMETERS <= 0 or billions is None:
        return False
    return int(billions) > settings.OLLAMA_MAX_PARAMETERS


def installed_models():
    try:
        answer = requests.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=5)
        answer.raise_for_status()
    except Exception as error:
        return [], str(error)

    models = []
    for entry in answer.json().get("models", []):
        if "name" not in entry:
            continue
        size = parameter_billions((entry.get("details") or {}).get("parameter_size"))
        models.append({"name": entry["name"], "size": size, "heavy": is_heavy(size)})

    models.sort(key=lambda entry: entry["name"])
    return models, None


def loaded_model():
    try:
        answer = requests.get(f"{settings.OLLAMA_HOST}/api/ps", timeout=5)
        answer.raise_for_status()
        running = answer.json().get("models", [])
        return running[0]["name"] if running else None
    except Exception:
        return None

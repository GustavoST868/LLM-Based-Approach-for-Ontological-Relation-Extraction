import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from common import console, language, settings


TRIPLET_TOKEN = "<triplet>"
SUBJECT_TOKEN = "<subj>"
OBJECT_TOKEN = "<obj>"

tokenizer = None
runner = None
device = None


def resolve_device():
    global device
    if device is not None:
        return device

    wanted = (settings.REBEL_DEVICE or "auto").strip().lower()
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if wanted == "auto":
        device = "cuda" if has_cuda else "cpu"
        if not has_cuda:
            console.once("rebel-cpu", "CUDA unavailable; REBEL will run on the CPU.")
    elif wanted.startswith("cuda") and not has_cuda:
        console.warn(f"REBEL_DEVICE='{settings.REBEL_DEVICE}' was asked for but CUDA is "
                     "unavailable; falling back to the CPU.")
        device = "cpu"
    else:
        device = wanted

    return device


def already_downloaded(model_name):
    folder = settings.MODEL_CACHE_DIR / ("models--" + model_name.replace("/", "--")) / "snapshots"
    if not folder.is_dir():
        return False
    return any(
        weight.suffix in (".safetensors", ".bin")
        for snapshot in folder.iterdir() if snapshot.is_dir()
        for weight in snapshot.iterdir()
    )


def load():
    global tokenizer, runner
    if runner is not None:
        return tokenizer, runner

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    name = settings.REBEL_MODEL
    where = resolve_device()
    arguments = {}

    if not os.path.isdir(name):
        settings.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        arguments["cache_dir"] = str(settings.MODEL_CACHE_DIR)
        if already_downloaded(name):
            arguments["local_files_only"] = True
        else:
            console.info(f"Downloading REBEL '{name}'…")

    tokenizer = AutoTokenizer.from_pretrained(name, **arguments)
    runner = AutoModelForSeq2SeqLM.from_pretrained(name, **arguments).to(where)
    runner.eval()
    console.detail(f"REBEL '{name}' ready on '{where}'.")
    return tokenizer, runner


def prepare():
    load()


def unload():
    global tokenizer, runner
    if runner is None:
        return

    import gc
    del runner, tokenizer
    runner = None
    tokenizer = None
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    console.detail("REBEL unloaded from memory.")


def sentence_windows(snippet):
    try:
        sentences = language.split_sentences(snippet)
    except Exception as error:
        console.once("rebel-split-fallback",
                     f"spaCy could not split the sentences for REBEL ({error}); using the whole "
                     "snippet as one window.", console.warn)
        sentences = [snippet]

    if not sentences:
        return [snippet]

    size = max(1, settings.REBEL_WINDOW_SENTENCES)
    stride = max(1, settings.REBEL_WINDOW_STRIDE)
    if len(sentences) <= size:
        return [" ".join(sentences)]

    windows = []
    start = 0
    while start < len(sentences):
        window = sentences[start:start + size]
        if not window:
            break
        windows.append(" ".join(window))
        if start + size >= len(sentences):
            break
        start += stride
    return windows


def extract_triplets(text):
    triplets = []
    subject, relation, target = "", "", ""
    current = None

    text = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text.split():
        if token == TRIPLET_TOKEN:
            if relation:
                triplets.append({"head": subject.strip(), "type": relation.strip(),
                                 "tail": target.strip()})
            subject, relation, target = "", "", ""
            current = "subject"
        elif token == SUBJECT_TOKEN:
            if relation:
                triplets.append({"head": subject.strip(), "type": relation.strip(),
                                 "tail": target.strip()})
            relation, target = "", ""
            current = "target"
        elif token == OBJECT_TOKEN:
            relation = ""
            current = "relation"
        elif current == "subject":
            subject += " " + token
        elif current == "target":
            target += " " + token
        elif current == "relation":
            relation += " " + token

    if subject and relation and target:
        triplets.append({"head": subject.strip(), "type": relation.strip(), "tail": target.strip()})
    return triplets


def triples_of_window(window):
    tokenizer, runner = load()
    import torch

    encoded = tokenizer(
        window, max_length=settings.REBEL_MAX_INPUT_TOKENS, truncation=True, return_tensors="pt"
    ).to(resolve_device())

    with torch.no_grad():
        generated = runner.generate(
            **encoded,
            max_length=settings.REBEL_MAX_OUTPUT_TOKENS,
            num_beams=settings.REBEL_BEAMS,
            num_return_sequences=min(settings.REBEL_RETURNED_SEQUENCES, settings.REBEL_BEAMS),
        )

    triples = []
    for sequence in tokenizer.batch_decode(generated, skip_special_tokens=False):
        triples.extend(extract_triplets(sequence))
    return triples


def triples_of(snippet):
    if not snippet or not snippet.strip():
        return []

    seen = set()
    found = []
    for window in sentence_windows(snippet):
        for triple in triples_of_window(window):
            key = (triple["head"].lower(), triple["type"].lower(), triple["tail"].lower())
            if not all(key) or key in seen:
                continue
            seen.add(key)
            triple["window"] = window
            found.append(triple)
            if settings.REBEL_MAX_TRIPLES and len(found) >= settings.REBEL_MAX_TRIPLES:
                return found
    return found

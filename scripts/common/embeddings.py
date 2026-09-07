import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from common import console, settings


COMPARED_FIELDS = ("description", "definition")

encoder = None
device = None


def resolve_device():
    global device
    if device is not None:
        return device

    wanted = (settings.EMBEDDING_DEVICE or "auto").strip().lower()
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if wanted == "auto":
        device = "cuda" if has_cuda else "cpu"
        if not has_cuda:
            console.once("embeddings-cpu", "CUDA unavailable; embeddings will run on the CPU.")
    elif wanted.startswith("cuda") and not has_cuda:
        console.warn(f"EMBEDDING_DEVICE='{settings.EMBEDDING_DEVICE}' was asked for but CUDA is "
                     "unavailable; falling back to the CPU.")
        device = "cpu"
    else:
        device = wanted

    return device


def batch_size():
    if settings.EMBEDDING_BATCH_SIZE > 0:
        return settings.EMBEDDING_BATCH_SIZE
    return 64 if resolve_device().startswith("cuda") else 16


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
    global encoder
    if encoder is not None:
        return encoder

    from sentence_transformers import SentenceTransformer

    name = settings.EMBEDDING_MODEL
    where = resolve_device()
    arguments = {"device": where}

    if not os.path.isdir(name):
        settings.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        arguments["cache_folder"] = str(settings.MODEL_CACHE_DIR)
        if already_downloaded(name):
            arguments["local_files_only"] = True
        else:
            console.info(f"Downloading the embedding model '{name}'…")

    encoder = SentenceTransformer(name, **arguments)
    console.detail(f"Embedding model '{name}' ready on '{where}'.")
    return encoder


def unload():
    global encoder
    if encoder is None:
        return

    import gc
    del encoder
    encoder = None
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    console.detail("Embedding model unloaded from memory.")


def encode(texts, prefix=""):
    import numpy

    model = load()
    if prefix:
        texts = [prefix + text for text in texts]

    vectors = model.encode(
        texts,
        batch_size=batch_size(),
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return numpy.asarray(vectors, dtype=numpy.float32)


def encode_documents(texts):
    return encode(texts)


def encode_queries(texts):
    return encode(texts, prefix=settings.EMBEDDING_QUERY_PREFIX)


def relation_text(relation):
    parts = [relation.get(field, "") for field in COMPARED_FIELDS]
    return " ".join(part.strip() for part in parts if part and part.strip())


def similarity_matrix(relations):
    vectors = encode_documents([relation_text(relation) for relation in relations])
    return vectors @ vectors.T

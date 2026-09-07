import hashlib
import json
import re

import numpy

from common import console, embeddings, papers, relations, settings


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}
MIN_PARAGRAPH_CHARACTERS = 80
MIN_SENTENCE_CHARACTERS = 20

PASSAGES_PER_CORPUS = 10
MIN_SCORE = 0.0

CHUNKING_VERSION = 2
VERSION_KEY = "__chunking_version__"
MODEL_KEY = "__embedding_model__"


def corpus_names():
    if not settings.RAG_DIR.exists():
        return []
    return sorted(
        entry.name for entry in settings.RAG_DIR.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )


def corpus_dir(corpus):
    return settings.RAG_DIR / corpus


def index_dir(corpus):
    return settings.RAG_INDEX_DIR / corpus.replace(" ", "_").lower()


def manifest_file(corpus):
    return index_dir(corpus) / "manifest.json"


def chunks_file(corpus):
    return index_dir(corpus) / "chunks.jsonl"


def vectors_file(corpus):
    return index_dir(corpus) / "embeddings.npy"


def source_files(corpus):
    root = corpus_dir(corpus)
    if not root.exists():
        return []

    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if settings.RAG_INDEX_DIR in path.parents:
            continue
        if path.parent == root and path.stem.upper() == "README":
            continue
        files.append(path)
    return files


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(corpus):
    path = manifest_file(corpus)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as source:
            return json.load(source)
    except (json.JSONDecodeError, OSError):
        return {}


def save_manifest(corpus, hashes):
    index_dir(corpus).mkdir(parents=True, exist_ok=True)
    manifest = dict(hashes)
    manifest[VERSION_KEY] = CHUNKING_VERSION
    manifest[MODEL_KEY] = settings.EMBEDDING_MODEL
    with manifest_file(corpus).open("w", encoding="utf-8") as target:
        json.dump(manifest, target, indent=2, ensure_ascii=False)


def load_chunks(corpus):
    path = chunks_file(corpus)
    if not path.exists():
        return []
    chunks = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def save_chunks(corpus, chunks):
    index_dir(corpus).mkdir(parents=True, exist_ok=True)
    with chunks_file(corpus).open("w", encoding="utf-8") as target:
        for chunk in chunks:
            target.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def load_vectors(corpus):
    path = vectors_file(corpus)
    if not path.exists():
        return None
    try:
        vectors = numpy.load(path)
    except (OSError, ValueError):
        return None
    return vectors if vectors.size else None


def save_vectors(corpus, vectors):
    index_dir(corpus).mkdir(parents=True, exist_ok=True)
    path = vectors_file(corpus)
    if vectors is None or len(vectors) == 0:
        path.unlink(missing_ok=True)
        return
    numpy.save(path, numpy.asarray(vectors, dtype=numpy.float32))


def paragraphs_of_file(path):
    if path.suffix.lower() == ".pdf":
        return [text for text, page in papers.paragraphs_of(path,
                                                            min_characters=MIN_PARAGRAPH_CHARACTERS)]
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as error:
        console.warn(f"RAG: could not read '{path}': {error}")
        return []

    paragraphs = []
    for block in re.split(r'\n\s*\n', raw):
        paragraph = " ".join(block.split())
        if len(paragraph) >= MIN_PARAGRAPH_CHARACTERS:
            paragraphs.append(paragraph)
    return paragraphs


def sentences_of_file(path):
    sentences = []
    for paragraph in paragraphs_of_file(path):
        for sentence in relations.sentences_of(paragraph):
            if len(sentence) >= MIN_SENTENCE_CHARACTERS:
                sentences.append(sentence)
    return sentences


def update_index(corpus):
    root = corpus_dir(corpus)
    files = source_files(corpus)
    manifest = load_manifest(corpus)

    if manifest.pop(VERSION_KEY, None) != CHUNKING_VERSION:
        if manifest:
            console.info(f"RAG[{corpus}]: chunking changed; reindexing every file.")
        manifest = {}

    indexed_model = manifest.pop(MODEL_KEY, None)
    if manifest and indexed_model != settings.EMBEDDING_MODEL:
        console.info(f"RAG[{corpus}]: embedding model changed ('{indexed_model}' -> "
                     f"'{settings.EMBEDDING_MODEL}'); recomputing the whole index.")
        manifest = {}

    if not files:
        if manifest or chunks_file(corpus).exists() or vectors_file(corpus).exists():
            console.info(f"RAG[{corpus}]: folder is empty; clearing the existing index.")
            save_manifest(corpus, {})
            save_chunks(corpus, [])
            save_vectors(corpus, None)
        else:
            console.detail(f"RAG[{corpus}]: no material; continuing without this corpus.")
        return [], None

    hashes = {str(path.relative_to(root)): file_hash(path) for path in files}
    changed = {name for name, digest in hashes.items() if manifest.get(name) != digest}
    removed = set(manifest) - set(hashes)

    chunks = load_chunks(corpus)
    vectors = load_vectors(corpus)
    healthy = vectors is not None and len(chunks) == len(vectors)

    if not changed and not removed and healthy:
        console.info(f"RAG[{corpus}]: index up to date — {len(chunks)} passage(s).")
        return chunks, vectors

    if not healthy:
        changed = set(hashes)
        removed = set()
        chunks, vectors = [], None

    stale = changed | removed
    kept_positions = [position for position, chunk in enumerate(chunks)
                      if chunk["source_file"] not in stale]
    kept_chunks = [chunks[position] for position in kept_positions]
    kept_vectors = vectors[kept_positions] if vectors is not None and kept_positions else None

    changed_files = [path for path in files if str(path.relative_to(root)) in changed]
    console.info(f"RAG[{corpus}]: {len(changed_files)} new/changed and {len(removed)} removed "
                 "file(s); updating the index.")

    new_chunks = []
    new_texts = []
    for path in changed_files:
        name = str(path.relative_to(root))
        sentences = sentences_of_file(path)
        console.detail(f"  {name}: {len(sentences)} passage(s)")
        for position, text in enumerate(sentences):
            new_chunks.append({"source_file": name, "chunk_index": position, "text": text})
            new_texts.append(text)

    new_vectors = None
    if new_texts:
        embeddings.load()
        console.show(step="Indexing the RAG corpora",
                     unit=f"{corpus}: {len(new_texts)} passages", force=True)
        new_vectors = embeddings.encode_documents(new_texts)

    if kept_vectors is not None and new_vectors is not None:
        all_vectors = numpy.concatenate([kept_vectors, new_vectors], axis=0)
    else:
        all_vectors = new_vectors if new_vectors is not None else kept_vectors

    all_chunks = kept_chunks + new_chunks

    save_manifest(corpus, hashes)
    save_chunks(corpus, all_chunks)
    save_vectors(corpus, all_vectors)

    console.info(f"RAG[{corpus}]: index updated — {len(all_chunks)} passage(s) from "
                 f"{len(files)} file(s).")
    return all_chunks, all_vectors


def update_all_indexes():
    corpora = corpus_names()
    if not corpora:
        console.info("RAG: no corpus folder in rag/; running without retrieved context.")
        return {}
    return {corpus: update_index(corpus) for corpus in corpora}


def closest_passages(texts, chunks, vectors, query_vectors):
    if not chunks or vectors is None or len(vectors) == 0:
        return [[] for text in texts]

    scores = query_vectors @ vectors.T

    results = []
    for row in scores:
        picked = []
        for position in numpy.argsort(-row)[:PASSAGES_PER_CORPUS]:
            score = float(row[position])
            if score < MIN_SCORE:
                continue
            chunk = chunks[position]
            picked.append({
                "text": chunk["text"],
                "source_file": chunk["source_file"],
                "score": score,
            })
        results.append(picked)
    return results


def retrieve(texts, indexes):
    if not texts:
        return []
    if not indexes:
        return [{} for text in texts]

    query_vectors = embeddings.encode_queries(texts)
    per_corpus = {
        corpus: closest_passages(texts, chunks, vectors, query_vectors)
        for corpus, (chunks, vectors) in indexes.items()
    }

    return [
        {corpus: results[position] for corpus, results in per_corpus.items()}
        for position in range(len(texts))
    ]

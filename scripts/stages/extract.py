import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (cleanup, console, embeddings, model, papers, prompts, rag,
                    relations, settings, storage)


PAUSE_BETWEEN_SNIPPETS = 1

MAX_READING_TOKENS = 320

MAX_VERBS = 8
MAX_LINKS = 6
MAX_READING_CHARS = 600

READING_LABEL = re.compile(r"^\W*(verbs|reading|links)\s*[:\-–]\s*(.*)$", re.IGNORECASE)
READING_LINK = re.compile(r"^([^|]+)\|([^|]+)\|([^|]+)$")

EMPTY_ANSWERS = ("none", "nenhum", "n/a", "-", "—", "")


reading_counts = {
    "readings": 0,
    "unusable": 0,
    "dropped_links": 0,
}

last_snippet = None
last_notes = ""


def reset_counts():
    relations.reset_counts()
    for key in reading_counts:
        reading_counts[key] = 0


def counts_summary():
    pairs = list(relations.counts_summary())
    if reading_counts["readings"]:
        pairs.append(("Snippet readings", reading_counts["readings"]))
    if reading_counts["unusable"]:
        pairs.append(("Readings with nothing usable", reading_counts["unusable"]))
    if reading_counts["dropped_links"]:
        pairs.append(("Links dropped (not in the snippet)", reading_counts["dropped_links"]))
    return pairs


def reading_is_on():
    return settings.current().use_reading


def notes_for(snippet):
    global last_snippet, last_notes

    if not reading_is_on() or not snippet or not snippet.strip():
        return ""

    if snippet == last_snippet:
        return last_notes

    last_snippet = snippet
    last_notes = read_snippet(snippet)
    return last_notes


def read_snippet(snippet):
    prompt = prompts.reading_prompt(snippet)

    reading_counts["readings"] += 1
    try:
        answer = model.ask(prompt, max_tokens=MAX_READING_TOKENS)
    except model.TruncatedAnswer as error:
        reading_counts["unusable"] += 1
        console.once("reading-truncated",
                     f"A snippet reading was cut off before any line arrived; extraction "
                     f"continues without notes. {error}", console.warn)
        return ""
    except Exception as error:
        reading_counts["unusable"] += 1
        console.once("reading-failed",
                     f"The snippet reading failed ({error}); extraction continues without notes.",
                     console.warn)
        return ""
    finally:
        model.count_operation("snippet reading")

    notes = notes_from_answer(answer, snippet)
    if not notes:
        reading_counts["unusable"] += 1
        console.detail(f"Nothing usable in the snippet reading; extracting without notes. "
                       f"It began: {str(answer)[:120]!r}")
    return notes


def labelled_lines(answer):
    found = {}
    for line in relations.without_reasoning(answer).splitlines():
        match = READING_LABEL.match(line.strip())
        if match:
            found.setdefault(match.group(1).lower(), match.group(2).strip())
    return found


def kept_verbs(raw, snippet):
    kept = []
    seen = set()

    for item in raw.split(";"):
        item = item.strip().strip('"').strip()
        if not item or item.lower() in EMPTY_ANSWERS:
            continue
        if not relations.is_in_source(item, snippet):
            continue
        if item.lower() in seen:
            continue
        seen.add(item.lower())
        kept.append(item)
        if len(kept) >= MAX_VERBS:
            break

    return kept


def kept_links(raw, snippet):
    kept = []
    seen = set()

    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry or entry.lower() in EMPTY_ANSWERS:
            continue

        match = READING_LINK.match(entry)
        if not match:
            reading_counts["dropped_links"] += 1
            continue

        subject, cue, target = (part.strip() for part in match.groups())
        if not (subject and cue and target):
            reading_counts["dropped_links"] += 1
            continue

        if not (relations.is_in_source(subject, snippet)
                and relations.is_in_source(target, snippet)):
            reading_counts["dropped_links"] += 1
            continue

        key = (subject.lower(), target.lower())
        if key in seen:
            continue
        seen.add(key)
        kept.append(f"{subject} | {cue} | {target}")
        if len(kept) >= MAX_LINKS:
            break

    return kept


def trimmed_reading(raw):
    text = " ".join(raw.split())
    if len(text) <= MAX_READING_CHARS:
        return text

    cut = text[:MAX_READING_CHARS]
    end = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[:end + 1] if end > 0 else cut.rsplit(" ", 1)[0]).strip()


def notes_from_answer(answer, snippet):
    lines = labelled_lines(answer)

    verbs = kept_verbs(lines.get("verbs", ""), snippet)
    reading = trimmed_reading(lines.get("reading", ""))
    links = kept_links(lines.get("links", ""), snippet)

    if not reading and not links:
        return ""

    block = []
    if verbs:
        block.append("Verbs: " + "; ".join(verbs))
    if reading:
        block.append("Reading: " + reading)
    block.append("Links: " + ("; ".join(links) if links else "none"))
    return "\n".join(block)


def record_source(found, capture, prompt, notes=""):
    for relation in found:
        relation["prompt"] = capture.get("prompt") or prompt
        relation["model_output"] = capture.get("answer")
        if notes:
            relation["snippet_reading"] = notes
    return found


def ask_for_class(snippet, category, passages, notes):
    relation_type = category[0]
    prompt = prompts.class_extraction_prompt(
        snippet, category, passages=passages, notes=notes,
    )

    capture = {}
    found = relations.ask_for_relations(
        prompt, expected_type=relation_type, source_text=snippet,
        kind="extraction", capture=capture,
    )
    return record_source(found, capture, prompt, notes)


def ask_openly(snippet, passages, notes):
    prompt = prompts.open_extraction_prompt(
        snippet, passages=passages, notes=notes,
    )

    capture = {}
    found = relations.ask_for_relations(
        prompt, expected_type=None, source_text=snippet, free_type=True,
        kind="extraction", capture=capture,
    )
    return record_source(found, capture, prompt, notes)


def relations_in(snippet, passages=None, on_request=None):
    if not snippet or not snippet.strip():
        return

    options = settings.current()
    categories = prompts.categories()

    extraction_requests = len(categories) if options.use_classes else 1
    reading_requests = 1 if reading_is_on() else 0
    total = extraction_requests + reading_requests

    if reading_requests and on_request is not None:
        on_request("reading the snippet", 1, total)
    notes = notes_for(snippet)

    if not options.use_classes:
        if on_request is not None:
            on_request("open extraction", total, total)
        yield from ask_openly(snippet, passages, notes)
        return

    for position, category in enumerate(categories, start=1):
        if on_request is not None:
            on_request(category[0], position + reading_requests, total)
        yield from ask_for_class(snippet, category, passages, notes)


def requests_per_snippet():
    options = settings.current()
    asked = len(prompts.categories()) if options.use_classes else 1
    return asked + (1 if reading_is_on() else 0)


def prepare_models(options):
    indexes = {}

    if options.use_rag:
        embeddings.load()
        indexes = rag.update_all_indexes()
        embeddings.unload()

    return indexes


def sentences_of(paragraph):
    kept = [text for text in relations.sentences_of(paragraph)
            if len(text) >= settings.MIN_SENTENCE_CHARACTERS]
    return kept or [paragraph]


def snippets_of(pdf_path, options):
    paragraphs = papers.paragraphs_of(pdf_path)
    if not options.by_sentence:
        return paragraphs

    return [
        (sentence, page_number)
        for paragraph, page_number in paragraphs
        for sentence in sentences_of(paragraph)
    ]


def context_for(texts, indexes):
    usable = {
        corpus: (chunks, vectors)
        for corpus, (chunks, vectors) in (indexes or {}).items()
        if chunks and vectors is not None
    }
    if not usable:
        return [{} for text in texts]

    console.detail(f"RAG: retrieving the top {rag.PASSAGES_PER_CORPUS} sentences per corpus "
                   f"({', '.join(usable)}) for {len(texts)} snippet(s).")
    passages = rag.retrieve(texts, usable)
    embeddings.unload()
    return passages


def rag_sources_of(passages):
    return [
        {"corpus": corpus, "file": item["source_file"], "score": round(item["score"], 3)}
        for corpus, items in (passages or {}).items()
        for item in items
    ]


def extract_pdf(pdf_path, position, pdf_count, raw_file, indexes, done, signature,
                total_so_far, started_at, options):
    snippets = snippets_of(pdf_path, options)
    if not snippets:
        console.warn(f"No text found in {pdf_path.name}; skipping it.")
        return total_so_far

    word = options.unit_word
    pending = [
        (number, text, page)
        for number, (text, page) in enumerate(snippets, start=1)
        if (pdf_path.name, number) not in done
    ]
    already_done = len(snippets) - len(pending)

    if not pending:
        console.info(f"{pdf_path.name}: all {len(snippets)} {word}(s) already done.")
        return total_so_far

    console.info(
        f"{pdf_path.name}: {len(snippets)} {word}(s)"
        + (f", {already_done} already done, {len(pending)} to do." if already_done else ".")
    )

    passages = context_for([text for number, text, page in pending], indexes)
    requests_total = requests_per_snippet()

    for place, (number, snippet, page_number) in enumerate(pending):
        console.show(
            document=pdf_path.name,
            document_position=f"{position}/{pdf_count}",
            paragraph=f"{number}/{len(snippets)} {word}",
            unit=f"0/{requests_total}",
            relations=total_so_far,
            elapsed=console.elapsed_since(started_at),
            force=True,
        )

        def show_request(label, request_number, request_total):
            console.show(unit=f"{request_number}/{request_total} {label}",
                         elapsed=console.elapsed_since(started_at))

        for relation in relations_in(snippet, passages=passages[place],
                                     on_request=show_request):
            relation.update(
                source_pdf=pdf_path.name,
                source_page=page_number,
                paragraph_index=number,
                level=options.unit,
                query_text=snippet,
                rag_sources=rag_sources_of(passages[place]),
            )
            raw_file.write(json.dumps(relation, ensure_ascii=False) + "\n")
            raw_file.flush()

            total_so_far += 1
            console.show(relations=total_so_far)

        storage.append_extraction_progress({
            "signature": signature,
            "source_pdf": pdf_path.name,
            "paragraph_index": number,
            "paragraph_total": len(snippets),
        })

        time.sleep(PAUSE_BETWEEN_SNIPPETS)

    return total_so_far


def load_checkpoint(options, signature):
    if options.restart:
        console.info("Starting from scratch: the previous checkpoint is discarded.")
        storage.reset_extraction_progress()
        return set(), False, False

    done, finished = storage.load_extraction_progress(signature)

    if finished and settings.RAW_FILE.exists():
        return done, False, True

    if done and settings.RAW_FILE.exists():
        dropped = storage.drop_unfinished_paragraphs(done)
        console.info(f"Resuming: {len(done)} {options.unit_word}(s) already in the checkpoint.")
        if dropped:
            console.detail(f"{dropped} line(s) from unfinished {options.unit_word}s were dropped "
                           f"from {settings.RAW_FILE.name} and will be extracted again.")
        return done, True, False

    if done and not settings.RAW_FILE.exists():
        console.warn(f"{settings.RAW_FILE.name} is gone; the checkpoint was discarded and "
                     "extraction restarts.")
    storage.reset_extraction_progress()
    return set(), False, False


def run(options=None):
    options = options or settings.current()
    signature = options.signature()

    if not settings.PAPERS_DIR.exists():
        console.error(f"{settings.PAPERS_DIR} not found.")
        return False

    pdf_files = sorted(settings.PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        console.error(f"No PDF found in {settings.PAPERS_DIR}.")
        return False

    settings.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    done, resuming, finished = load_checkpoint(options, signature)

    if finished:
        console.info(f"Extraction already complete for this configuration: "
                     f"{storage.count_lines(settings.RAW_FILE)} relation(s) in "
                     f"{settings.RAW_FILE.name}. Reusing them and moving on.")
        console.detail("Use --restart to extract again.")
        return True

    reset_counts()
    started_at = time.time()
    found_total = 0

    try:
        indexes = prepare_models(options)
        console.info(f"Extracting from {len(pdf_files)} PDF(s), "
                     f"one request per {options.unit_word}…")

        with settings.RAW_FILE.open("a" if resuming else "w", encoding="utf-8") as raw_file:
            for position, pdf_path in enumerate(pdf_files, start=1):
                found_total = extract_pdf(
                    pdf_path, position, len(pdf_files), raw_file, indexes, done, signature,
                    found_total, started_at, options,
                )
    finally:
        console.clear_status()
        cleanup.release_everything()

    storage.append_extraction_progress({"signature": signature, "event": "finished"})

    console.success(f"Extraction finished in {console.elapsed_since(started_at)}.")
    console.summary([
        ("Relations this run", found_total),
        ("Relations in file", storage.count_lines(settings.RAW_FILE)),
        *counts_summary(),
        ("Saved to", settings.RAW_FILE.name),
    ])
    return True


if __name__ == "__main__":
    from common import command_line
    sys.exit(command_line.run_single_stage(run, "extract"))

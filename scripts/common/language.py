import subprocess
import sys

from common import console, settings


VERB_TAGS = {"VERB", "AUX"}

SUBJECT_LINKS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
OBJECT_LINKS = {"dobj", "obj", "attr", "oprd", "acomp", "dative", "xcomp"}
PREPOSITION_LINKS = {"prep", "agent"}
CLAUSE_LINKS = {"acl", "relcl", "advcl", "pcomp"}

MAX_SKELETON_LINES = 12
MAX_LINKS_PER_VERB = 3

loaded = {}


def download_model(name):
    try:
        import spacy
        spacy.load(name)
    except OSError:
        console.info(f"spaCy model '{name}' not found; downloading it now…")
        subprocess.run([sys.executable, "-m", "spacy", "download", name], check=True)
    except ImportError:
        raise RuntimeError("spaCy is not installed. Install it with: pip install spacy")


def pipeline(name=None):
    name = name or settings.SPACY_MODEL
    if name not in loaded:
        download_model(name)
        import spacy
        loaded[name] = spacy.load(name)
    return loaded[name]


def prepare():
    pipeline()
    console.detail(f"Linguistic model '{settings.SPACY_MODEL}' ready.")


def unload():
    if not loaded:
        return

    import gc
    loaded.clear()
    gc.collect()
    console.detail("Linguistic model unloaded from memory.")


def split_sentences(text):
    return [sentence.text.strip() for sentence in pipeline()(text).sents if sentence.text.strip()]


def noun_phrase(token):
    for chunk in token.doc.noun_chunks:
        if chunk.start <= token.i < chunk.end:
            return chunk.text
    return token.text


def extended_phrase(token):
    phrase = noun_phrase(token)
    for child in token.children:
        if child.dep_ != "prep":
            continue
        for grandchild in child.children:
            if grandchild.dep_ == "pobj":
                return f"{phrase} {child.text} {noun_phrase(grandchild)}"
    return phrase


def verb_connections(verb):
    subject = None
    objects = []
    prepositional = []

    for child in verb.children:
        if child.dep_ in SUBJECT_LINKS and subject is None:
            subject = noun_phrase(child)
        elif child.dep_ in OBJECT_LINKS:
            objects.append(("", extended_phrase(child)))
        elif child.dep_ in PREPOSITION_LINKS:
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    prepositional.append((child.text, noun_phrase(grandchild)))

    if subject is None and verb.dep_ in CLAUSE_LINKS:
        subject = noun_phrase(verb.head)

    return subject, (objects + prepositional)[:MAX_LINKS_PER_VERB]


def verb_skeleton(text):
    if not text or not text.strip():
        return ""

    lines = []
    for token in pipeline()(text):
        if token.pos_ not in VERB_TAGS:
            continue
        if token.dep_ in ("aux", "auxpass"):
            continue

        subject, links = verb_connections(token)
        if not subject or not links:
            continue

        auxiliaries = " ".join(child.text for child in token.lefts
                               if child.dep_ in ("aux", "auxpass"))
        verb = f"{auxiliaries} {token.text}".strip()
        tied = ", ".join(f"{label} {phrase}".strip() for label, phrase in links)

        lines.append(f"- {subject} — {verb} — {tied}")
        if len(lines) >= MAX_SKELETON_LINES:
            break

    return "\n".join(lines)


def annotate(text):
    skeleton = verb_skeleton(text)
    if not skeleton:
        return text
    return f"{text}\n\nVerbs and what they connect:\n{skeleton}"

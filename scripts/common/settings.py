import os
from dataclasses import dataclass, asdict, field
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = SCRIPTS_DIR.parent

PAPERS_DIR = BASE_DIR / "papers"
RAG_DIR = BASE_DIR / "rag"
RAG_INDEX_DIR = RAG_DIR / ".index"

PROMPTS_DIR = BASE_DIR / "prompts"
PROMPT_FILE = PROMPTS_DIR / "prompt.txt"
PROMPT_THINKING_FILE = PROMPTS_DIR / "prompt_thinking.txt"
PROMPT_EXTRACTION_FILE = PROMPTS_DIR / "prompt_extraction.txt"
PROMPT_FILES = (PROMPT_THINKING_FILE, PROMPT_EXTRACTION_FILE, PROMPT_FILE)

MODEL_CACHE_DIR = BASE_DIR / "model"

RESULT_DIR = BASE_DIR / "result"
RAW_FILE = RESULT_DIR / "relations_raw.jsonl"
MERGED_FILE = RESULT_DIR / "relations_merged.json"
DEDUPLICATED_FILE = RESULT_DIR / "relations_deduplicated.json"
FINAL_FILE = RESULT_DIR / "relations_extracted.json"
CLUSTERS_DIR = RESULT_DIR / "clusters"
CLUSTER_NAMES_FILE = RESULT_DIR / "cluster_names.json"
ISOLATED_FILE = CLUSTERS_DIR / "isolated_relations.json"

EXTRACT_PROGRESS_FILE = RESULT_DIR / "extraction_progress.jsonl"
DISAMBIGUATE_PROGRESS_FILE = RESULT_DIR / "disambiguation_progress.json"

SNAPSHOTS_DIR = RESULT_DIR / "models"

WEB_DIR = SCRIPTS_DIR / "web"
WEB_INDEX_FILE = WEB_DIR / "index.html"


def text_setting(name, default):
    value = (os.getenv(name) or "").strip()
    return value or default


def number_setting(name, default):
    value = (os.getenv(name) or "").strip()
    try:
        return float(value) if value else default
    except ValueError:
        return default


def whole_number_setting(name, default):
    return int(number_setting(name, default))


def flag_setting(name, default):
    value = (os.getenv(name) or "").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return default


HF_TOKEN = os.getenv("HF_TOKEN") or None
SEED = 42
VERBOSE = flag_setting("VERBOSE", False)


OLLAMA_HOST = text_setting("OLLAMA_HOST", "http://localhost:11434")

OLLAMA_MODEL = text_setting("OLLAMA_EXTRACTION_MODEL", "gemma3:12b")

OLLAMA_MAX_PARAMETERS = number_setting("OLLAMA_MAX_PARAMETERS", 12)

OLLAMA_KEEP_ALIVE = text_setting("OLLAMA_KEEP_ALIVE", "30m")

CONTEXT_SIZE = whole_number_setting("OLLAMA_NUM_CTX", 0)

CONTEXT_MARGIN = whole_number_setting("OLLAMA_CTX_MARGIN", 2048)

CONTEXT_LIMIT = whole_number_setting("OLLAMA_CTX_MAX", 0)
CONTEXT_LIMIT_FALLBACK = whole_number_setting("OLLAMA_CTX_MAX_FALLBACK", 32768)

CHARS_PER_TOKEN = number_setting("OLLAMA_CHARS_PER_TOKEN", 3.0)

MAX_ANSWER_TOKENS = whole_number_setting("OLLAMA_NUM_PREDICT", -1)

RECYCLE_EVERY = whole_number_setting("LLM_RECYCLE_EVERY", 100)
RECYCLE_PAUSE = number_setting("LLM_RECYCLE_PAUSE", 3)


EMBEDDING_MODEL = text_setting("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
EMBEDDING_DEVICE = text_setting("EMBEDDING_DEVICE", "auto")
EMBEDDING_BATCH_SIZE = whole_number_setting("EMBEDDING_BATCH_SIZE", 0)

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def default_query_prefix(model_name):
    name = (model_name or "").lower()
    if "bge" in name and "-en" in name:
        return BGE_QUERY_INSTRUCTION
    return ""


EMBEDDING_QUERY_PREFIX = os.getenv("EMBEDDING_QUERY_PREFIX", default_query_prefix(EMBEDDING_MODEL))

SIMILARITY_THRESHOLD = number_setting("RELATION_SIMILARITY_THRESHOLD", 0.80)

MIN_SENTENCE_CHARACTERS = whole_number_setting("MIN_SENTENCE_CHARACTERS", 30)

DASHBOARD_PORT = whole_number_setting("DASHBOARD_PORT", 8765)


STAGES = (
    ("extract", "Extract relations from the PDFs in papers/"),
    ("merge", "Merge duplicate relations (same class, similar name)"),
    ("cluster", "Group relations by semantic similarity"),
    ("disambiguate", "Reduce semantic redundancy inside each cluster"),
    ("merge_global", "Enforce one relation per name across every class"),
)
STAGE_KEYS = tuple(key for key, label in STAGES)
STAGE_LABELS = dict(STAGES)

UNIT_PARAGRAPH = "paragraph"
UNIT_SENTENCE = "sentence"
UNITS = (UNIT_PARAGRAPH, UNIT_SENTENCE)


def valid_choice(value, allowed, default):
    cleaned = (value or "").strip().lower()
    return cleaned if cleaned in allowed else default


@dataclass
class Options:

    model: str = OLLAMA_MODEL
    unit: str = UNIT_PARAGRAPH
    use_rag: bool = True
    use_classes: bool = True
    use_reading: bool = True
    restart: bool = False
    stages: list = field(default_factory=lambda: list(STAGE_KEYS))

    @classmethod
    def from_environment(cls):
        return cls(
            model=text_setting("OLLAMA_EXTRACTION_MODEL", OLLAMA_MODEL),
            unit=valid_choice(os.getenv("EXTRACTION_UNIT"), UNITS, UNIT_PARAGRAPH),
            use_rag=flag_setting("USE_RAG", True),
            use_classes=flag_setting("USE_ONTOLOGY_CLASSES", True),
            use_reading=flag_setting("USE_SNIPPET_ANALYSIS", True),
            restart=flag_setting("EXTRACTION_FRESH", False),
            stages=list(STAGE_KEYS),
        )

    def as_environment(self):
        return {
            "OLLAMA_EXTRACTION_MODEL": self.model,
            "EXTRACTION_UNIT": self.unit,
            "USE_RAG": "1" if self.use_rag else "0",
            "USE_ONTOLOGY_CLASSES": "1" if self.use_classes else "0",
            "USE_SNIPPET_ANALYSIS": "1" if self.use_reading else "0",
            "EXTRACTION_FRESH": "1" if self.restart else "0",
        }

    def apply(self):
        os.environ.update(self.as_environment())
        use(self)
        return self

    def as_dict(self):
        return asdict(self)

    @property
    def by_sentence(self):
        return self.unit == UNIT_SENTENCE

    @property
    def unit_word(self):
        return "sentence" if self.by_sentence else "paragraph"

    def signature(self):
        name = self.model

        marks = []
        if self.by_sentence:
            marks.append("sentences")
        if not self.use_rag:
            marks.append("norag")
        if not self.use_classes:
            marks.append("openclasses")
        if not self.use_reading:
            marks.append("noreading")

        return f"{name}[{'+'.join(marks)}]" if marks else name

    def describe(self):
        return [
            ("Model", self.model),
            ("Extraction unit", "one request per " + self.unit_word),
            ("RAG context", "on" if self.use_rag else "off"),
            ("Ontology classes", "on" if self.use_classes else "off"),
            ("Snippet reading", "on" if self.use_reading else "off"),
            ("Start from scratch", "yes" if self.restart else "no (resume checkpoint)"),
            ("Stages", ", ".join(self.stages)),
        ]


chosen = None


def current():
    global chosen
    if chosen is None:
        chosen = Options.from_environment()
    return chosen


def use(options):
    global chosen
    chosen = options
    return options

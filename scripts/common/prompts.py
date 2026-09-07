import re
from string import Template

from common import settings


SECTION_HEADER = re.compile(r"^@@@[ \t]*([\w.-]+)[ \t]*@@@[ \t]*$")
OPTIONAL_BLOCK = re.compile(r"\{\{#(\w+)\}\}(.*?)\{\{/\1\}\}", re.DOTALL)

CLASS_HEADER = re.compile(r"^\$\$\s*(.+?)\s*\$\$\s*$")
CLASS_FIELD_HEADER = re.compile(r"^\[(definition|example|narrative)\]\s*$")
CLASS_FIELDS = ("definition", "example", "narrative")
CATALOGUE_SECTION = "relation_categories"

CORPUS_LABELS = {
    "Paper Ontology": "Ontology corpus — most related sentences",
    "Paper Geology": "Geology corpus — most related sentences",
}


sections_cache = {}
sections_origin = {}
files_state = None


def files_signature():
    signature = []
    for path in settings.PROMPT_FILES:
        try:
            info = path.stat()
        except OSError as error:
            raise FileNotFoundError(f"Could not read the prompt file {path}: {error}") from None
        signature.append((info.st_mtime_ns, info.st_size))
    return tuple(signature)


def split_sections(text):
    found = {}
    name = None
    body = []

    def close():
        if name is not None:
            found[name] = "\n".join(body).strip("\n") + "\n"

    for line in text.splitlines():
        header = SECTION_HEADER.match(line)
        if header:
            close()
            name = header.group(1)
            body = []
            continue
        if name is not None:
            body.append(line)

    close()
    return found


def read_all_files():
    merged = {}
    origin = {}

    for path in settings.PROMPT_FILES:
        for name, body in split_sections(path.read_text(encoding="utf-8")).items():
            if name in merged:
                raise ValueError(
                    f"The section '@@@ {name} @@@' is defined twice: in {origin[name].name} and in "
                    f"{path.name}. A section belongs to one file only."
                )
            merged[name] = body
            origin[name] = path

    return merged, origin


def sections():
    global sections_cache, sections_origin, files_state

    signature = files_signature()
    if signature != files_state:
        sections_cache, sections_origin = read_all_files()
        files_state = signature
        forget_categories()
    return sections_cache


def section(name):
    available = sections()
    if name not in available:
        files = ", ".join(path.name for path in settings.PROMPT_FILES)
        raise KeyError(
            f"There is no '@@@ {name} @@@' section in {files}. "
            f"They hold: {', '.join(available) or 'no section at all'}."
        )
    return available[name]


def render(name, active_blocks=(), **values):
    text = section(name)
    text = OPTIONAL_BLOCK.sub(
        lambda match: match.group(2) if match.group(1) in active_blocks else "",
        text,
    )
    return Template(text).substitute(**values)


categories_cache = None


def forget_categories():
    global categories_cache
    categories_cache = None


def parse_categories(text):
    source = f"the '@@@ {CATALOGUE_SECTION} @@@' section of {settings.PROMPT_FILE.name}"
    found = []
    name = None
    fields = {}
    field = None

    def close():
        if name is None:
            return
        missing = [key for key in CLASS_FIELDS if not fields.get(key)]
        if missing:
            raise ValueError(f"Class '{name}' in {source} is missing: {', '.join(missing)}.")
        found.append([
            name,
            " ".join(fields["definition"]).strip(),
            " ".join(fields["example"]).strip(),
            " ".join(fields["narrative"]).strip(),
        ])

    for line in text.splitlines():
        header = CLASS_HEADER.match(line)
        if header:
            close()
            name = header.group(1)
            fields = {key: [] for key in CLASS_FIELDS}
            field = None
            continue

        field_header = CLASS_FIELD_HEADER.match(line)
        if field_header:
            if name is None:
                raise ValueError(f"Field '{line.strip()}' before any '$$ class $$' in {source}.")
            field = field_header.group(1)
            continue

        if field is not None and line.strip():
            fields[field].append(line.strip())

    close()

    if not found:
        raise ValueError(f"No relation class found in {source}.")

    names = [entry[0] for entry in found]
    duplicates = {value for value in names if names.count(value) > 1}
    if duplicates:
        raise ValueError(f"Duplicate classes in {source}: {', '.join(sorted(duplicates))}.")

    return found


def categories():
    global categories_cache
    text = section(CATALOGUE_SECTION)
    if categories_cache is None:
        categories_cache = parse_categories(text)
    return categories_cache


def category_names():
    return {name for name, definition, example, narrative in categories()}


def format_context(passages):
    if not passages:
        return ""

    blocks = []
    for corpus, items in passages.items():
        if not items:
            continue
        label = CORPUS_LABELS.get(corpus, corpus or "Reference corpus")
        lines = "\n".join(f"- {item['text']}" for item in items)
        blocks.append(f"{label} ({len(items)}):\n{lines}")

    return "\n\n".join(blocks)


def format_relations(relations):
    return "\n\n".join(
        f"Relation {position + 1}:\n"
        f"Name: {relation.get('name', '')}\n"
        f"Description: {relation.get('description', '')}\n"
        f"Definition: {relation.get('definition', '')}\n"
        f"Type: {relation.get('relation_type', '')}\n"
        f"Usage: {relation.get('usage_example', '')}"
        for position, relation in enumerate(relations)
    )


def format_other_classes(relation_type):
    return "\n".join(
        f"- {name}"
        for name, definition, example, narrative in categories()
        if name != relation_type
    )


def active_blocks(context, notes):
    blocks = []
    if context:
        blocks.append("context")
    if notes:
        blocks.append("analysis")
    return tuple(blocks)


def reading_prompt(snippet):
    return render("analyse_snippet", chunk_text=snippet)


def class_extraction_prompt(snippet, category, passages=None, notes=""):
    relation_type, definition, example, narrative = category
    context = format_context(passages)
    return render(
        "extract",
        active_blocks=active_blocks(context, notes),
        relation_type=relation_type,
        relation_definition=definition,
        examples_narrative=narrative,
        other_categories=format_other_classes(relation_type),
        snippets=context,
        analysis=notes,
        chunk_text=snippet,
    )


def open_extraction_prompt(snippet, passages=None, notes=""):
    context = format_context(passages)
    return render(
        "extract_open",
        active_blocks=active_blocks(context, notes),
        snippets=context,
        analysis=notes,
        chunk_text=snippet,
    )


def merge_prompt(relations):
    return render(
        "merge",
        count=len(relations),
        listed=format_relations(relations),
        relation_type=relations[0]["relation_type"],
    )


def global_merge_prompt(relations):
    return render(
        "merge_global",
        count=len(relations),
        listed=format_relations(relations),
    )


def disambiguation_prompt(first, second):
    return render(
        "disambiguate",
        name1=first["name"],
        relation_type1=first["relation_type"],
        description1=first["description"],
        definition1=first.get("definition", ""),
        usage_example1=first["usage_example"],
        name2=second["name"],
        relation_type2=second["relation_type"],
        description2=second["description"],
        definition2=second.get("definition", ""),
        usage_example2=second["usage_example"],
    )

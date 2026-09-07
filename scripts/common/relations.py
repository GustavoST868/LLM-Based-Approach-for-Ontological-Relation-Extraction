import difflib
import json
import re
import time
from collections import defaultdict

from common import console, model, prompts, settings


REQUIRED_FIELDS = ("name", "description", "definition", "relation_type", "usage_example")
MERGEABLE_FIELDS = ("name", "description", "definition", "relation_type", "usage_example")

NAME_PATTERN = re.compile(r"^[a-z]+(?:_[a-z]+){0,2}$")

MAX_DESCRIPTION_SENTENCES = 1
MAX_DESCRIPTION_CHARS = 250
MAX_DEFINITION_SENTENCES = 3
MAX_DEFINITION_CHARS = 400

PLACEHOLDER_WORDS = {"subject", "predicate", "object", "entity", "entity a", "entity b"}

MERGE_BATCH_SIZE = 10

NOTHING = object()


counts = {
    "requests": 0,
    "unreadable": 0,
    "truncated": 0,
    "off_candidate": 0,
}


def reset_counts():
    for key in counts:
        counts[key] = 0


def counts_summary():
    pairs = [("Model requests", counts["requests"])]
    if counts["unreadable"]:
        pairs.append(("Unreadable answers", counts["unreadable"]))
    if counts["truncated"]:
        pairs.append(("Truncated answers", counts["truncated"]))
    if counts["off_candidate"]:
        pairs.append(("Answers about another candidate", counts["off_candidate"]))
    return pairs


def plain_text(text):
    if not text:
        return ""
    kept = "".join(c for c in str(text).lower() if c.isalnum() or c.isspace())
    return " ".join(kept.split())


def word_text(text):
    return re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower())


def snake_name(name):
    if not isinstance(name, str):
        return ""
    cleaned = name.strip().lower().replace(" ", "_").replace("-", "_")
    cleaned = re.sub(r"[^a-z_]", "", cleaned)
    words = [word for word in cleaned.split("_") if word]
    return "_".join(words[:3])


def sentences_of(text, split_on_semicolon=False):
    pattern = r"(?<=[.!?])\s+|;\s+" if split_on_semicolon else r"(?<=[.!?])\s+"
    parts = re.split(pattern, str(text or "").strip())
    return [part.strip() for part in parts if part.strip()]


def condense(text, max_sentences, max_chars, split_on_semicolon=False):
    sentences = sentences_of(text, split_on_semicolon=split_on_semicolon)
    if not sentences:
        return ""

    kept = sentences[0]
    for sentence in sentences[1:max_sentences]:
        if len(kept) + 1 + len(sentence) > max_chars:
            break
        kept = f"{kept} {sentence}"

    if len(kept) <= max_chars:
        return kept

    window = kept[:max_chars]
    cut = ""
    for separator in ("; ", ", "):
        position = window.rfind(separator)
        if position >= max_chars * 0.6:
            cut = window[:position]
            break
    if not cut:
        cut = window.rsplit(" ", 1)[0]

    cut = cut.rstrip(" ,;:-—")
    return cut if cut.endswith((".", "!", "?")) else f"{cut}."


def tidy_usage_example(value):
    if not isinstance(value, str):
        return ""

    value = value.strip()
    if value.count("|") == 2:
        return value

    separator = r'([|–—,;]\s*|\s+[-–—]\s+|\s*,\s*|\s*;\s*)'
    match = re.match(rf'^\s*(.+?)\s*{separator}\s*(.+?)\s*{separator}\s*(.+?)\s*$', value)
    if match:
        parts = [match.group(1).strip(), match.group(3).strip(), match.group(5).strip()]
        if all(parts):
            return " | ".join(parts)
    return value


def predicate_matches_name(predicate, name, threshold=0.6):
    normalized = snake_name(predicate)
    if not normalized or not name:
        return False
    if normalized == name:
        return True
    return difflib.SequenceMatcher(None, normalized, name).ratio() >= threshold


def is_placeholder(phrase):
    return plain_text(phrase) in PLACEHOLDER_WORDS


def is_in_source(phrase, source_text, min_overlap=0.6):
    phrase_words = word_text(phrase).strip()
    source = word_text(source_text)
    if not phrase_words:
        return False
    if phrase_words in source:
        return True

    long_words = [word for word in phrase_words.split() if len(word) > 2]
    if not long_words:
        return True

    source_words = set(source.split())
    hits = sum(1 for word in long_words if word in source_words)
    return (hits / len(long_words)) >= min_overlap


def same_entity(phrase, other, threshold=0.75):
    left = word_text(phrase).strip()
    right = word_text(other).strip()
    if not left or not right:
        return False
    if left in right or right in left:
        return True
    return difflib.SequenceMatcher(None, left, right).ratio() >= threshold


def rejected(reason, candidate):
    console.detail(f"rejected ({reason}): {str(candidate)[:160]}")


def check_relation(candidate, expected_type=None, check_usage=True, source_text=None,
                   free_type=False):
    if not isinstance(candidate, dict):
        rejected("not a dict", candidate)
        return None

    missing = set(REQUIRED_FIELDS) - set(candidate)
    if missing:
        rejected(f"missing fields: {', '.join(sorted(missing))}", candidate)
        return None

    relation = {}
    for field in REQUIRED_FIELDS:
        value = candidate.get(field)
        if not isinstance(value, str) or not value.strip():
            rejected(f"empty field '{field}'", candidate)
            return None
        relation[field] = " ".join(value.split())

    relation["description"] = condense(
        relation["description"], MAX_DESCRIPTION_SENTENCES, MAX_DESCRIPTION_CHARS,
        split_on_semicolon=True,
    )
    relation["definition"] = condense(
        relation["definition"], MAX_DEFINITION_SENTENCES, MAX_DEFINITION_CHARS
    )
    if not relation["description"] or not relation["definition"]:
        rejected("description or definition empty after condensing", candidate)
        return None

    relation["name"] = snake_name(relation["name"])
    if not relation["name"] or not NAME_PATTERN.fullmatch(relation["name"]):
        rejected(f"invalid name: {relation['name']}", candidate)
        return None

    if expected_type:
        relation["relation_type"] = expected_type
    elif free_type:
        relation["relation_type"] = relation["relation_type"].strip()
    else:
        written = relation["relation_type"].strip()
        if written not in prompts.category_names():
            rejected(f"unknown class: {written}", candidate)
            return None
        relation["relation_type"] = written

    relation["usage_example"] = tidy_usage_example(relation["usage_example"])
    if not check_usage:
        return relation

    if relation["usage_example"].count("|") != 2:
        rejected(f"usage example is not 'Subject | Predicate | Object': "
                 f"{relation['usage_example']}", candidate)
        return None

    subject, predicate, target = (part.strip() for part in relation["usage_example"].split("|"))

    if not predicate_matches_name(predicate, relation["name"]):
        rejected(f"predicate '{predicate}' does not match the name '{relation['name']}'", candidate)
        return None

    if is_placeholder(subject) or is_placeholder(target):
        rejected(f"usage example uses a schema placeholder: {relation['usage_example']}", candidate)
        return None

    if source_text and (not is_in_source(subject, source_text)
                        or not is_in_source(target, source_text)):
        rejected(f"usage example is not in the source text: {relation['usage_example']}", candidate)
        return None

    return relation


def validate(candidate, free_type=False):
    return check_relation(candidate, expected_type=None, check_usage=True, free_type=free_type)


def read_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return NOTHING


def means_no_relations(value):
    if value is None:
        return True
    if isinstance(value, list):
        return len(value) == 0
    if isinstance(value, dict):
        if not value:
            return True
        if isinstance(value.get("relations"), list) and not value["relations"]:
            return True
        return all(item in (None, [], "", {}) for item in value.values())
    return False


def find_relation_list(value):
    if isinstance(value, list):
        return value

    if isinstance(value, dict):
        if isinstance(value.get("relations"), list):
            return value["relations"]
        if set(REQUIRED_FIELDS).issubset(value.keys()):
            return [value]
        lists = [item for item in value.values() if isinstance(item, list)]
        if len(lists) == 1:
            return lists[0]
        return [] if means_no_relations(value) else None

    return [] if means_no_relations(value) else None


def balanced_end(text, start):
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    inside_string = False
    escaped = False

    for index in range(start, len(text)):
        character = text[index]
        if escaped:
            escaped = False
            continue
        if character == "\\" and inside_string:
            escaped = True
            continue
        if character == '"':
            inside_string = not inside_string
            continue
        if inside_string:
            continue
        if character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def json_values_in(text):
    index = 0
    while index < len(text):
        if text[index] in "{[":
            end = balanced_end(text, index)
            if end is not None:
                try:
                    value = json.loads(text[index:end])
                except json.JSONDecodeError:
                    pass
                else:
                    yield value
                    index = end
                    continue
        index += 1


def looks_like_relations(value):
    if isinstance(value, list):
        return True
    if isinstance(value, dict):
        if isinstance(value.get("relations"), list):
            return True
        return set(REQUIRED_FIELDS).issubset(value.keys())
    return False


def json_from_text(text):
    text = re.sub(r'```json\s*|\s*```', '', text)
    values = list(json_values_in(text))
    if not values:
        return None
    for value in reversed(values):
        if looks_like_relations(value):
            return value
    return values[-1]


REASONING_BLOCK = re.compile(
    r"<\s*(think|thinking|reasoning|analysis)\s*>.*?<\s*/\s*\1\s*>", re.DOTALL | re.IGNORECASE
)
REASONING_OPEN = re.compile(r"<\s*(think|thinking|reasoning|analysis)\s*>", re.IGNORECASE)
REASONING_CLOSE = re.compile(r"<\s*/\s*(think|thinking|reasoning|analysis)\s*>", re.IGNORECASE)


def without_reasoning(answer):
    if not answer:
        return ""

    text = REASONING_BLOCK.sub("", answer)

    closings = list(REASONING_CLOSE.finditer(text))
    if closings:
        text = text[closings[-1].end():]

    openings = list(REASONING_OPEN.finditer(text))
    if openings:
        text = text[openings[-1].end():]

    return text.strip()


def parse_answer(answer, expected_type=None, want_list=False, check_schema=True,
                 source_text=None, free_type=False):
    answer = without_reasoning(answer)
    if answer.startswith("```"):
        answer = answer.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    parsed = read_json(answer)
    if parsed is NOTHING:
        parsed = json_from_text(answer)
    if parsed is None or parsed is NOTHING:
        rejected("could not read any JSON", answer[:200])
        return None

    if not want_list:
        if isinstance(parsed, dict) and isinstance(parsed.get("relations"), list) \
                and len(parsed["relations"]) == 1:
            parsed = parsed["relations"][0]
        if not isinstance(parsed, dict):
            return None
        if not check_schema:
            return parsed
        return check_relation(parsed, expected_type, check_usage=True,
                              source_text=source_text, free_type=free_type)

    items = find_relation_list(parsed)
    if not isinstance(items, list):
        rejected("the answer is not a list of relations", items)
        return None

    found = []
    for item in items:
        relation = check_relation(item, expected_type, check_usage=check_schema,
                                  source_text=source_text, free_type=free_type)
        if relation is not None:
            found.append(relation)
    return found


def ask_model(prompt, expected_type=None, want_list=False, check_schema=True, source_text=None,
              free_type=False, kind="request", capture=None, attempts=3, pause=2,
              max_tokens=None):
    empty = [] if want_list else None
    answer = None

    def record(attempt):
        if capture is not None:
            capture["prompt"] = prompt
            capture["answer"] = answer
            capture["attempts"] = attempt + 1

    try:
        for attempt in range(attempts):
            try:
                counts["requests"] += 1
                answer = model.ask(prompt, max_tokens=max_tokens)
                record(attempt)

                result = parse_answer(
                    answer, expected_type, want_list=want_list, check_schema=check_schema,
                    source_text=source_text, free_type=free_type,
                )
                if result is not None:
                    return result

                counts["unreadable"] += 1
                console.detail(
                    "No relation could be read from the answer; moving on to the next request. "
                    f"It began: {str(answer)[:120]!r}"
                )
                return empty

            except model.TruncatedAnswer as error:
                record(attempt)
                counts["truncated"] += 1
                console.once(
                    "truncated-answer",
                    f"A model answer was cut off before the end. Identical retries are skipped "
                    f"(temperature 0, fixed seed). {error}",
                    console.warn,
                )
                return empty

            except Exception as error:
                console.warn(f"Request failed (attempt {attempt + 1}/{attempts}): {error}")
                if attempt < attempts - 1:
                    time.sleep(pause)

        return empty
    finally:
        model.count_operation(kind)


def ask_for_relations(prompt, expected_type=None, source_text=None, free_type=False,
                      kind="extraction", capture=None):
    return ask_model(
        prompt, expected_type=expected_type, want_list=True, check_schema=True,
        source_text=source_text, free_type=free_type, kind=kind, capture=capture,
    )


def ask_for_answer(prompt, kind="request"):
    return ask_model(prompt, want_list=False, check_schema=False, kind=kind, attempts=2, pause=1)


def name_key(text):
    return " ".join(str(text).lower().split())


def similar_names(first, second, threshold=0.85):
    if first == second:
        return True
    if first == second + "s" or second == first + "s":
        return True
    return difflib.SequenceMatcher(None, first, second).ratio() >= threshold


def group_by_name(relations, same_class_required):
    groups = defaultdict(list)
    keys = []

    for relation in relations:
        relation_type = name_key(relation.get("relation_type", ""))
        name = name_key(relation.get("name", ""))

        found = None
        for existing_type, existing_name in keys:
            if same_class_required and existing_type != relation_type:
                continue
            if similar_names(existing_name, name):
                found = (existing_type, existing_name)
                break

        if found is None:
            found = (relation_type, name)
            keys.append(found)
        groups[found].append(relation)

    return groups


def read_merge_groups(answer, batch):
    if not isinstance(answer, dict):
        return None

    groups = answer.get("groups")
    if not isinstance(groups, list) or not groups:
        return None

    seen = set()
    parsed = []

    for group in groups:
        if not isinstance(group, dict):
            return None

        members = group.get("members")
        if not isinstance(members, list) or not members:
            return None

        try:
            indexes = [int(member) - 1 for member in members]
        except (TypeError, ValueError):
            return None

        if any(index < 0 or index >= len(batch) for index in indexes):
            return None
        if seen.intersection(indexes):
            return None

        seen.update(indexes)
        parsed.append((indexes, group))

    return parsed if len(seen) == len(batch) else None


def merge_batch(batch, build_prompt, tally):
    if len(batch) == 1:
        return list(batch)

    free_type = not settings.current().use_classes
    answer = ask_for_answer(build_prompt(batch), kind="merge")
    parsed = read_merge_groups(answer, batch)

    if parsed is None:
        console.detail(f"The merge answer did not partition the {len(batch)} relations; "
                       "keeping them apart.")
        tally["refused"] += 1
        return list(batch)

    kept = []
    for indexes, fields in parsed:
        if len(indexes) == 1:
            kept.append(batch[indexes[0]])
            continue

        merged = validate({field: fields.get(field) for field in MERGEABLE_FIELDS},
                          free_type=free_type)
        if merged is None:
            console.detail(f"A merge of {len(indexes)} relations failed validation; "
                           "keeping them apart.")
            tally["refused"] += 1
            kept.extend(batch[index] for index in indexes)
            continue

        names = [batch[index].get("name") for index in indexes]
        console.detail(f"Merged {len(indexes)} relations into '{merged['name']}': {names}")
        tally["merged"] += len(indexes) - 1

        relation = dict(batch[indexes[0]])
        relation.update(merged)
        kept.append(relation)

    return kept


def merge_one_group(group, build_prompt, tally):
    current = list(group)
    while len(current) > 1:
        following = []
        for start in range(0, len(current), MERGE_BATCH_SIZE):
            following.extend(merge_batch(current[start:start + MERGE_BATCH_SIZE],
                                         build_prompt, tally))
        if len(following) >= len(current):
            return following
        current = following

    return current


def fold_exact_duplicates(relations, by_class=True):
    kept = {}
    order = []

    for relation in relations:
        name = name_key(relation.get("name", ""))
        key = (name_key(relation.get("relation_type", "")), name) if by_class else name

        if key not in kept:
            kept[key] = relation
            order.append(key)
            continue

        existing = kept[key]
        for field in ("description", "definition"):
            candidate = (relation.get(field) or "").strip()
            current = (existing.get(field) or "").strip()
            if candidate and (not current or len(candidate) < len(current)):
                existing[field] = candidate

    return [kept[key] for key in order]


def merge_by_name(relations, same_class_required, build_prompt, label):
    groups = group_by_name(relations, same_class_required=same_class_required)
    candidates = [group for group in groups.values() if len(group) > 1]
    total_before = len(relations)

    tally = {"merged": 0, "refused": 0}
    kept = [group[0] for group in groups.values() if len(group) == 1]

    for position, group in enumerate(candidates, start=1):
        console.show(
            step=label,
            unit=f"group {position}/{len(candidates)} · '{group[0].get('name')}'",
            relations=total_before - tally["merged"],
        )
        kept.extend(merge_one_group(group, build_prompt, tally))

    before_folding = len(kept)
    kept = fold_exact_duplicates(kept, by_class=same_class_required)
    folded = before_folding - len(kept)

    summary = [
        ("Relations in", total_before),
        ("Name groups examined", len(candidates)),
        ("Merged away", total_before - len(kept)),
    ]
    if folded:
        summary.append(("Folded as exact duplicates", folded))
    if tally["refused"]:
        summary.append(("Merges refused", tally["refused"]))
    summary.append(("Relations out", len(kept)))

    return kept, summary

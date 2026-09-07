import html
import re
import shutil
import subprocess
from collections import Counter, namedtuple
from pathlib import Path

from PyPDF2 import PdfReader

from common import console


REFERENCES_TITLE = (
    r'references(?:\s+(?:cited|and\s+notes))?|reference\s+list|cited\s+references|'
    r'bibliography|works\s+cited|literature\s+cited|'
    r'referências(?:\s+bibliográficas)?|referencias(?:\s+bibliograficas)?|bibliografia'
)
SECTION_NUMBER = r'(?:\d+(?:\.\d+)*|[IVXivx]+)[.)]?\s+'

REFERENCES_HEADING = re.compile(
    r'(?i)^[ \t]*(?:' + SECTION_NUMBER + r')?(?:' + REFERENCES_TITLE + r')[ \t]*[:.]?[ \t]*$',
    re.MULTILINE,
)

REFERENCES_AFTER_GUTTER = re.compile(
    r'(?i)^.*?[ \t]{3,}((?:' + SECTION_NUMBER + r')?(?:' + REFERENCES_TITLE + r')[ \t]*[:.]?)[ \t]*$',
    re.MULTILINE,
)

REFERENCES_INLINE = re.compile(
    r'(?i)^(?:' + SECTION_NUMBER + r')?(?:' + REFERENCES_TITLE + r')\s*[:.]?\s+(?=\S)'
)

INTRODUCTION_TITLE = r'introduction|introdução|introducao|introducción|introduccion'

INTRODUCTION_HEADING = re.compile(
    r'(?i)^(?:' + SECTION_NUMBER + r')?(?:' + INTRODUCTION_TITLE + r')\s*[:.]?\s*$'
)
INTRODUCTION_NUMBERED_INLINE = re.compile(
    r'(?i)^(?:' + SECTION_NUMBER + r')(?:' + INTRODUCTION_TITLE + r')\s*[:.]?\s+(?=\S)'
)
INTRODUCTION_CAPS_INLINE = re.compile(
    r'^(?:INTRODUCTION|INTRODUÇÃO|INTRODUCAO|INTRODUCCIÓN|INTRODUCCION)\s*[:.]?\s+(?=\S)'
)

MAX_INTRODUCTION_PAGE = 3


COLUMN_MAX_CROSSING = 0.015
COLUMN_MIN_SIDE = 0.25
COLUMN_MIN_WORDS = 40
COLUMN_STEPS = 200
MAX_COLUMN_SPLITS = 2

LINE_OVERLAP_TOLERANCE = 0.5
PARAGRAPH_GAP_FACTOR = 0.5
FIRST_LINE_INDENT_FACTOR = 0.6
JUSTIFIED_SHARE = 0.8
SHORT_LINE_SHARE = 0.92

MARGIN_BAND_SHARE = 0.12
MIN_RUNNING_PAGES = 3

SENTENCE_END = re.compile(r'[.!?]["\')\]”’]*$')
CONTINUATION_CHARACTERS = ';,)]'

LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
}
LIGATURE_PATTERN = re.compile("[" + "".join(LIGATURES) + "]")

Word = namedtuple("Word", "x_min x_max y_min y_max text")
Line = namedtuple("Line", "x_min x_max y_min y_max text words")


def run_pdftotext(pdf_path, *flags):
    program = shutil.which("pdftotext")
    if not program:
        return None
    try:
        finished = subprocess.run(
            [program, *flags, str(pdf_path), "-"], check=True, capture_output=True, text=True
        )
        return finished.stdout
    except (OSError, subprocess.CalledProcessError) as error:
        console.warn(f"Error extracting text from {pdf_path}: {error}")
        return None


def pages_with_pypdf(pdf_path):
    pages = []
    try:
        with Path(pdf_path).open("rb") as source:
            reader = PdfReader(source)
            for page in reader.pages:
                pages.append(page.extract_text() or "")
    except Exception as error:
        console.warn(f"Error extracting text from {pdf_path}: {error}")
    return pages


def text_pages(pdf_path):
    output = run_pdftotext(pdf_path, "-layout")
    if output is not None:
        return output.split("\f")
    console.once("no-pdftotext",
                 "pdftotext not found; falling back to PyPDF2 text extraction.", console.warn)
    return pages_with_pypdf(pdf_path)


def expand_ligatures(text):
    return LIGATURE_PATTERN.sub(lambda match: LIGATURES[match.group(0)], text)


PAGE_TAG = re.compile(r'<page\b[^>]*>(.*?)</page>', re.S)
WORD_TAG = re.compile(r'<word\s+([^>]*)>(.*?)</word>', re.S)
ATTRIBUTE = re.compile(r'([\w-]+)="([^"]*)"')


def word_pages(pdf_path):
    output = run_pdftotext(pdf_path, "-bbox-layout")
    if not output:
        return None

    pages = []
    for page_body in PAGE_TAG.findall(output):
        words = []
        for attributes, text in WORD_TAG.findall(page_body):
            text = html.unescape(text).strip()
            if not text:
                continue
            values = dict(ATTRIBUTE.findall(attributes))
            try:
                words.append(Word(
                    float(values["xMin"]), float(values["xMax"]),
                    float(values["yMin"]), float(values["yMax"]),
                    expand_ligatures(text),
                ))
            except (KeyError, ValueError):
                continue
        pages.append(words)

    return pages or None


def find_column_cut(words):
    if len(words) < COLUMN_MIN_WORDS:
        return None

    left_edge = min(word.x_min for word in words)
    right_edge = max(word.x_max for word in words)
    span = right_edge - left_edge
    if span <= 0:
        return None

    best_cut = None
    best_share = COLUMN_MAX_CROSSING

    for step in range(COLUMN_STEPS + 1):
        cut = left_edge + span * (0.25 + 0.5 * step / COLUMN_STEPS)
        crossing = left = right = 0
        for word in words:
            if word.x_max <= cut:
                left += 1
            elif word.x_min >= cut:
                right += 1
            else:
                crossing += 1

        if min(left, right) < COLUMN_MIN_SIDE * len(words):
            continue
        share = crossing / len(words)
        if share < best_share:
            best_cut, best_share = cut, share

    return best_cut


def split_into_columns(words, depth=0):
    cut = find_column_cut(words) if depth < MAX_COLUMN_SPLITS else None
    if cut is None:
        return [words]

    left = [word for word in words if word.x_max <= cut]
    right = [word for word in words if word.x_min >= cut]

    for word in words:
        if word.x_max > cut > word.x_min:
            if cut - word.x_min >= word.x_max - cut:
                left.append(word)
            else:
                right.append(word)

    return split_into_columns(left, depth + 1) + split_into_columns(right, depth + 1)


def median(values):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def group_into_lines(words):
    if not words:
        return []

    line_height = median([word.y_max - word.y_min for word in words]) or 1.0
    tolerance = LINE_OVERLAP_TOLERANCE * line_height

    rows = []
    current = []
    current_center = None

    for word in sorted(words, key=lambda item: (item.y_min, item.x_min)):
        center = (word.y_min + word.y_max) / 2
        if current and abs(center - current_center) > tolerance:
            rows.append(current)
            current = []
        if not current:
            current_center = center
        current.append(word)

    if current:
        rows.append(current)

    lines = []
    for row in rows:
        row.sort(key=lambda item: item.x_min)
        lines.append(Line(
            min(word.x_min for word in row),
            max(word.x_max for word in row),
            min(word.y_min for word in row),
            max(word.y_max for word in row),
            " ".join(word.text for word in row),
            row,
        ))
    return lines


def group_into_paragraphs(lines):
    if not lines:
        return []

    line_height = median([line.y_max - line.y_min for line in lines]) or 1.0
    gaps = [following.y_min - current.y_max for current, following in zip(lines, lines[1:])]
    body_gap = median([gap for gap in gaps if gap >= 0]) if gaps else 0.0

    left_edges = Counter(round(line.x_min) for line in lines)
    body_left = min(edge for edge, count in left_edges.items() if count == max(left_edges.values()))
    body_right = max(line.x_max for line in lines)
    column_width = body_right - body_left

    full_lines = sum(1 for line in lines
                     if line.x_max >= body_left + SHORT_LINE_SHARE * column_width)
    justified = column_width > 0 and full_lines / len(lines) >= JUSTIFIED_SHARE

    paragraphs = []
    current = []

    for position, line in enumerate(lines):
        if current:
            previous = lines[position - 1]
            gap = line.y_min - previous.y_max
            starts_paragraph = (
                gap > body_gap + PARAGRAPH_GAP_FACTOR * line_height
                or line.x_min > body_left + FIRST_LINE_INDENT_FACTOR * line_height
                or (justified and previous.x_max < body_left + SHORT_LINE_SHARE * column_width)
            )
            if starts_paragraph:
                paragraphs.append(current)
                current = []
        current.append(line)

    if current:
        paragraphs.append(current)
    return paragraphs


def inline_hyphenations(texts):
    found = set()
    for text in texts:
        for match in re.finditer(r'\w+-\w+', text):
            found.add(match.group(0).lower())
    return found


def join_lines(texts, hyphenations=frozenset()):
    joined = ""
    for text in texts:
        text = " ".join(text.split())
        if not text:
            continue
        if not joined:
            joined = text
            continue

        tail = re.search(r'(\w+)-$', joined)
        head = re.match(r'(\w+)', text)
        if tail and head and text[:1].islower():
            compound = f"{tail.group(1)}-{head.group(1)}".lower()
            joined = joined + text if compound in hyphenations else joined[:-1] + text
        else:
            joined += " " + text

    return joined


def continues_previous(previous, following):
    if not previous or not following:
        return False
    if previous.endswith("-"):
        return True
    if SENTENCE_END.search(previous):
        return False
    return following[:1].islower() or following[:1] in CONTINUATION_CHARACTERS


def join_continuations(blocks):
    joined = []
    for text, page_number in blocks:
        if joined and continues_previous(joined[-1][0], text):
            previous, previous_page = joined[-1]
            merged = previous[:-1] + text if previous.endswith("-") else f"{previous} {text}"
            joined[-1] = (merged, previous_page)
            continue
        joined.append((text, page_number))
    return joined


def running_form(text):
    return re.sub(r'\d+', '#', " ".join(text.split())).lower()


def find_running_lines(pages_of_words):
    if len(pages_of_words) < MIN_RUNNING_PAGES:
        return set()

    seen = {}
    for page_number, words in enumerate(pages_of_words, start=1):
        if not words:
            continue
        top = min(word.y_min for word in words)
        bottom = max(word.y_max for word in words)
        band = MARGIN_BAND_SHARE * (bottom - top)

        for line in group_into_lines(words):
            in_margin = line.y_max <= top + band or line.y_min >= bottom - band
            if in_margin and line.text.strip():
                seen.setdefault(running_form(line.text), set()).add(page_number)

    threshold = max(MIN_RUNNING_PAGES, len(pages_of_words) // 2)
    return {text for text, pages in seen.items() if len(pages) >= threshold}


def is_header_or_footer(text):
    lowered = text.lower()
    return (
        "journal of south american earth sciences" in lowered
        or lowered.startswith("contents lists available")
        or lowered.startswith("journal homepage:")
        or re.fullmatch(r"\d+", text.strip()) is not None
    )


def clean_page(text):
    kept = [line for line in text.splitlines() if not is_header_or_footer(line.strip())]
    return "\n".join(kept)


def references_position(page_text):
    positions = []

    at_line_start = REFERENCES_HEADING.search(page_text)
    if at_line_start:
        positions.append(at_line_start.start())

    after_gutter = REFERENCES_AFTER_GUTTER.search(page_text)
    if after_gutter:
        positions.append(after_gutter.start(1))

    return min(positions) if positions else None


def cut_at_references(pages):
    kept = []
    for page_text in pages:
        start = references_position(page_text)
        if start is not None:
            truncated = page_text[:start]
            if truncated.strip():
                kept.append(truncated)
            break
        kept.append(page_text)
    return kept


def is_references_heading(text):
    return REFERENCES_HEADING.match(text.strip()) is not None


def after_introduction_heading(text):
    stripped = text.strip()
    if not stripped:
        return None

    if INTRODUCTION_HEADING.match(stripped):
        return ""

    for pattern in (INTRODUCTION_NUMBERED_INLINE, INTRODUCTION_CAPS_INLINE):
        match = pattern.match(stripped)
        if match:
            return stripped[match.end():].strip()

    return None


def drop_front_matter(blocks):
    for position, (text, page_number) in enumerate(blocks):
        if page_number > MAX_INTRODUCTION_PAGE:
            break

        body = after_introduction_heading(text)
        if body is None:
            continue

        remaining = blocks[position + 1:]
        return ([(body, page_number)] if body else []) + remaining

    console.warn(
        f"Introduction heading not found in the first {MAX_INTRODUCTION_PAGE} page(s); "
        "the text before the introduction was kept."
    )
    return blocks


def drop_references(blocks):
    for position, (text, page_number) in enumerate(blocks):
        stripped = text.strip()
        if position == 0:
            continue
        if is_references_heading(stripped) or REFERENCES_INLINE.match(stripped):
            return blocks[:position]
    return blocks


def without_running_lines(words, running_lines):
    kept = []
    for line in group_into_lines(words):
        text = line.text.strip()
        if is_header_or_footer(text) or running_form(text) in running_lines:
            continue
        kept.extend(line.words)
    return kept


def blocks_from_words(pages_of_words):
    collected = []
    reached_references = False
    running_lines = find_running_lines(pages_of_words)

    for page_number, words in enumerate(pages_of_words, start=1):
        if reached_references:
            break

        for column in split_into_columns(without_running_lines(words, running_lines)):
            for paragraph in group_into_paragraphs(group_into_lines(column)):
                texts = [line.text for line in paragraph]
                if any(is_references_heading(text) for text in texts):
                    reached_references = True
                    break
                collected.append((texts, page_number))
            if reached_references:
                break

    hyphenations = inline_hyphenations(text for texts, page in collected for text in texts)
    return [(join_lines(texts, hyphenations), page_number) for texts, page_number in collected]


def blocks_from_text(pages):
    cleaned = cut_at_references([expand_ligatures(clean_page(page)) for page in pages])
    hyphenations = inline_hyphenations(cleaned)

    blocks = []
    for page_number, page_text in enumerate(cleaned, start=1):
        for raw_paragraph in re.split(r'\n\s*\n', page_text):
            paragraph = join_lines(raw_paragraph.splitlines(), hyphenations)
            if paragraph:
                blocks.append((paragraph, page_number))
    return blocks


def keep_body_only(blocks):
    before = len(blocks)
    blocks = drop_references(drop_front_matter(blocks))
    dropped = before - len(blocks)
    if dropped:
        console.detail(f"Dropped {dropped} block(s) outside the paper body (front matter and "
                       f"references); {len(blocks)} kept.")
    return blocks


def paragraphs_of(pdf_path, min_characters=80):
    pages_of_words = word_pages(pdf_path)
    if pages_of_words:
        blocks = blocks_from_words(pages_of_words)
    else:
        pages = text_pages(pdf_path)
        if not pages:
            return []
        blocks = blocks_from_text(pages)

    blocks = keep_body_only(join_continuations(blocks))

    return [
        (paragraph, page_number)
        for paragraph, page_number in blocks
        if len(paragraph) >= min_characters
    ]

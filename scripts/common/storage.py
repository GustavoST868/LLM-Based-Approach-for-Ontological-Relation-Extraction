import json
import os
import shutil
import time

from common import settings


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        json.dump(data, target, indent=2, ensure_ascii=False)


def save_json_safely(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        json.dump(data, target, indent=2, ensure_ascii=False)
    temporary.replace(path)


def load_json(path, default=None):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as source:
            return json.load(source)
    except (json.JSONDecodeError, OSError):
        return default


def load_lines(path):
    items = []
    if not path.exists():
        return items

    with path.open("r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def count_lines(path):
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def missing_final_newline(path):
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as probe:
        probe.seek(-1, os.SEEK_END)
        return probe.read(1) != b"\n"


def append_line(entry, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as target:
        if missing_final_newline(path):
            target.write("\n")
        target.write(json.dumps(entry, ensure_ascii=False) + "\n")
        target.flush()


def load_extraction_progress(signature):
    done = set()
    finished = False
    path = settings.EXTRACT_PROGRESS_FILE
    if not path.exists():
        return done, finished

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return set(), False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue

        if entry.get("signature") != signature:
            return set(), False
        if entry.get("event") == "finished":
            finished = True
        elif entry.get("source_pdf") and entry.get("paragraph_index") is not None:
            done.add((entry["source_pdf"], entry["paragraph_index"]))

    return done, finished


def append_extraction_progress(entry):
    append_line(entry, settings.EXTRACT_PROGRESS_FILE)


def reset_extraction_progress():
    settings.EXTRACT_PROGRESS_FILE.unlink(missing_ok=True)


def drop_unfinished_paragraphs(done):
    raw_file = settings.RAW_FILE
    if not raw_file.exists():
        return 0

    temporary = raw_file.with_suffix(".jsonl.tmp")
    dropped = 0

    with raw_file.open("r", encoding="utf-8") as source, \
            temporary.open("w", encoding="utf-8") as target:
        for line in source:
            line = line.strip()
            if not line:
                continue
            try:
                relation = json.loads(line)
            except ValueError:
                dropped += 1
                continue

            if (relation.get("source_pdf"), relation.get("paragraph_index")) in done:
                target.write(line + "\n")
            else:
                dropped += 1

    temporary.replace(raw_file)
    return dropped


def clear_clusters():
    settings.CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
    for old in settings.CLUSTERS_DIR.glob("cluster_*.json"):
        old.unlink()
    settings.ISOLATED_FILE.unlink(missing_ok=True)


def save_cluster(cluster_id, relations):
    save_json(relations, settings.CLUSTERS_DIR / f"cluster_{cluster_id}.json")


def load_clusters():
    settings.CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
    clusters = {}
    for cluster_file in sorted(settings.CLUSTERS_DIR.glob("cluster_*.json")):
        cluster_id = int(cluster_file.stem.split("_")[1])
        clusters[cluster_id] = load_json(cluster_file, [])
    return clusters


def save_isolated(relations):
    save_json(relations, settings.ISOLATED_FILE)


def load_isolated():
    return load_json(settings.ISOLATED_FILE, [])


SNAPSHOT_FILES = (
    settings.RAW_FILE,
    settings.MERGED_FILE,
    settings.DEDUPLICATED_FILE,
    settings.FINAL_FILE,
    settings.CLUSTER_NAMES_FILE,
)


def folder_name_for(signature):
    name = (signature or "unknown_run").strip().replace("/", "__").replace("\\", "__")
    name = "".join(character for character in name if character.isprintable())
    return name or "unknown_run"


def save_snapshot(signature, options_summary=None):
    target = settings.SNAPSHOTS_DIR / folder_name_for(signature)
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)

    for source in SNAPSHOT_FILES:
        if source.is_file():
            shutil.copy2(source, target / source.name)

    if settings.CLUSTERS_DIR.exists() and any(settings.CLUSTERS_DIR.iterdir()):
        shutil.copytree(settings.CLUSTERS_DIR, target / "clusters", dirs_exist_ok=True)

    save_json(
        {"run": signature, "saved_at": time.time(), "options": options_summary or {}},
        target / "run_meta.json",
    )
    return target

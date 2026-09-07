import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import console, prompts, relations, settings, storage


def merge_duplicates(found):
    return relations.merge_by_name(
        found,
        same_class_required=True,
        build_prompt=prompts.merge_prompt,
        label="Merging duplicates",
    )


def run(options=None):
    options = options or settings.current()

    if not settings.RAW_FILE.exists():
        console.error(f"{settings.RAW_FILE.name} not found — run the 'extract' stage first.")
        return False

    found = storage.load_lines(settings.RAW_FILE)
    if not found:
        console.error(f"{settings.RAW_FILE.name} is empty; there is nothing to merge.")
        return False

    relations.reset_counts()
    console.info(f"Merging duplicates among {len(found)} raw relation(s)…")

    merged, summary = merge_duplicates(found)
    storage.save_json(merged, settings.MERGED_FILE)

    console.clear_status()
    console.success("Duplicate merge finished.")
    console.summary([*summary, *relations.counts_summary(),
                     ("Saved to", settings.MERGED_FILE.name)])
    return True


if __name__ == "__main__":
    from common import command_line
    sys.exit(command_line.run_single_stage(run, "merge"))

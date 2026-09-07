import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import console, prompts, relations, settings, storage


def merge_across_classes(found):
    return relations.merge_by_name(
        found,
        same_class_required=False,
        build_prompt=prompts.global_merge_prompt,
        label="Global merge",
    )


def run(options=None):
    options = options or settings.current()

    if not settings.DEDUPLICATED_FILE.exists():
        console.error(f"{settings.DEDUPLICATED_FILE.name} not found — run the 'disambiguate' "
                      "stage first.")
        return False

    found = storage.load_json(settings.DEDUPLICATED_FILE, [])
    if not found:
        console.error(f"{settings.DEDUPLICATED_FILE.name} is empty; there is nothing to merge.")
        return False

    relations.reset_counts()
    console.info(f"Enforcing one relation per name across {len(found)} relation(s)…")

    final, summary = merge_across_classes(found)
    storage.save_json(final, settings.FINAL_FILE)

    console.clear_status()
    console.success(f"Pipeline result: {len(final)} unique relation(s).")
    console.summary([*summary, *relations.counts_summary(),
                     ("Saved to", settings.FINAL_FILE.name)])
    return True


if __name__ == "__main__":
    from common import command_line
    sys.exit(command_line.run_single_stage(run, "merge_global"))

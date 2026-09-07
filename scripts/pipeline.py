import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import cleanup, command_line, console, settings, storage


def stage_module(key):
    return importlib.import_module(f"stages.{key}")


def normalize(keys):
    if not keys:
        return list(settings.STAGE_KEYS)
    wanted = set(keys)
    return [key for key in settings.STAGE_KEYS if key in wanted]


def unknown_stages(keys):
    return [key for key in keys or [] if key not in settings.STAGE_KEYS]


def run_stages(options, snapshot=True):
    stages = normalize(options.stages)

    console.heading("Run configuration")
    console.summary(options.describe())

    done = []
    try:
        for position, key in enumerate(stages, start=1):
            console.stage_banner(position, len(stages), settings.STAGE_LABELS.get(key, key))
            if not stage_module(key).run(options):
                console.error(f"Stage '{key}' could not be completed; stopping the pipeline.")
                return False
            done.append(key)
    finally:
        console.clear_status()
        cleanup.release_everything()

    if snapshot and done:
        target = storage.save_snapshot(options.signature(), dict(options.describe()))
        console.info(f"Results saved to {target}.")

    return True


def main(argv=None):
    parser = command_line.build_parser("Extract ontology relations from the papers in papers/.")
    parser.add_argument(
        "stages", nargs="*", metavar="stage",
        help="Stages to run, in any order (default: all of them). "
             f"One or more of: {', '.join(settings.STAGE_KEYS)}.",
    )
    arguments = parser.parse_args(argv)

    unknown = unknown_stages(arguments.stages)
    if unknown:
        console.error(f"Unknown stage(s): {', '.join(unknown)}. "
                      f"Valid: {', '.join(settings.STAGE_KEYS)}.")
        return 2

    options = command_line.options_from(arguments, stages=normalize(arguments.stages))

    if run_stages(options):
        console.success("Pipeline complete.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

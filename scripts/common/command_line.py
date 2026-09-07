import argparse
import os

from common import settings


def build_parser(description):
    parser = argparse.ArgumentParser(description=description)
    defaults = settings.Options.from_environment()

    parser.add_argument(
        "--model", default=defaults.model,
        help=f"Ollama model used by every stage (default: {defaults.model}).",
    )
    parser.add_argument(
        "--unit", choices=list(settings.UNITS), default=defaults.unit,
        help="How much text one extraction request is about (default: %(default)s).",
    )
    parser.add_argument(
        "--no-rag", dest="use_rag", action="store_false", default=defaults.use_rag,
        help="Do not inject passages from the corpora in rag/ into the prompts.",
    )
    parser.add_argument(
        "--open-classes", dest="use_classes", action="store_false", default=defaults.use_classes,
        help="Extract in one open request per snippet, with the model naming the relation type, "
             "instead of one request per class of the catalogue.",
    )
    parser.add_argument(
        "--no-reading", dest="use_reading", action="store_false", default=defaults.use_reading,
        help="Skip the reading request that precedes the extraction of each snippet.",
    )
    parser.add_argument(
        "--restart", action="store_true", default=defaults.restart,
        help="Discard the extraction checkpoint and extract every snippet again.",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=settings.VERBOSE,
        help="Print the diagnostic detail as well.",
    )
    return parser


def options_from(arguments, stages=None):
    if arguments.verbose:
        os.environ["VERBOSE"] = "1"
        settings.VERBOSE = True

    options = settings.Options(
        model=arguments.model,
        unit=arguments.unit,
        use_rag=arguments.use_rag,
        use_classes=arguments.use_classes,
        use_reading=arguments.use_reading,
        restart=arguments.restart,
        stages=list(stages) if stages else list(settings.STAGE_KEYS),
    )
    return options.apply()


def run_single_stage(stage_function, name):
    from common import cleanup, console

    parser = build_parser(f"Run the '{name}' stage on its own.")
    options = options_from(parser.parse_args(), stages=[name])

    console.heading(f"Stage: {name}")
    console.summary(options.describe())

    try:
        succeeded = stage_function(options)
    finally:
        console.clear_status()
        cleanup.release_everything()

    return 0 if succeeded else 1

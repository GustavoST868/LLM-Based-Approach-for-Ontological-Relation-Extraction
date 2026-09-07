import os
import sys
import time

from common import settings


RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"

STATUS_INTERVAL = 0.1

output = sys.stdout
on_terminal = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

status_text = ""
status_fields = {}
status_drawn = False
status_time = 0.0
announced = set()


def colored(text, color):
    return f"{color}{text}{RESET}" if on_terminal else text


def terminal_width():
    try:
        return max(40, os.get_terminal_size().columns - 1)
    except OSError:
        return 100


def shortened(text, width):
    single_line = text.replace("\n", " ")
    if len(single_line) <= width:
        return single_line
    return single_line[:width - 1] + "…"


def erase_status():
    global status_drawn
    if status_drawn and on_terminal:
        output.write("\r\033[2K")
        status_drawn = False


def draw_status():
    global status_drawn
    if status_text and on_terminal:
        output.write("\r\033[2K" + shortened(status_text, terminal_width()))
        output.flush()
        status_drawn = True


def write(text):
    erase_status()
    output.write(text + "\n")
    output.flush()
    draw_status()


def info(message):
    write(str(message))


def detail(message):
    if settings.VERBOSE:
        write(colored(str(message), DIM))


def success(message):
    write(colored(f"✔ {message}", GREEN))


def warn(message):
    write(colored(f"! {message}", YELLOW))


def error(message):
    write(colored(f"✖ {message}", RED))


def once(key, message, level=info):
    if key in announced:
        return False
    announced.add(key)
    level(message)
    return True


def heading(title):
    write("")
    write(colored(f"── {title} ", BOLD) + colored("─" * max(0, 60 - len(title)), DIM))


def stage_banner(position, total, label):
    write("")
    write(colored(f"[{position}/{total}] {label}", BOLD + CYAN))
    show(step=label, force=True)


def summary(pairs):
    if not pairs:
        return
    width = max(len(str(key)) for key, value in pairs)
    for key, value in pairs:
        write(f"  {colored(str(key).ljust(width), DIM)}  {value}")


def show(force=False, **fields):
    global status_time, status_text

    status_fields.update({key: value for key, value in fields.items() if value is not None})

    now = time.time()
    if not force and (now - status_time) < STATUS_INTERVAL:
        return
    status_time = now

    status_text = format_status(status_fields)
    draw_status()


def format_status(fields):
    parts = []
    if fields.get("step"):
        parts.append(fields["step"])

    document = fields.get("document")
    if document:
        position = fields.get("document_position")
        parts.append(f"{document} ({position})" if position else document)

    if fields.get("paragraph"):
        parts.append(f"par {fields['paragraph']}")
    if fields.get("unit"):
        parts.append(fields["unit"])

    found = fields.get("relations")
    if found is not None:
        parts.append(f"{found} relations" if isinstance(found, int) else str(found))

    if fields.get("elapsed"):
        parts.append(fields["elapsed"])

    return "  ·  ".join(str(part) for part in parts)


def clear_status():
    global status_text, status_fields
    erase_status()
    status_text = ""
    status_fields = {}


def elapsed_since(started_at):
    return time.strftime("%H:%M:%S", time.gmtime(max(0.0, time.time() - started_at)))

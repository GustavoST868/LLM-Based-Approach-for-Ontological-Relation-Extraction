import ctypes
import gc

from common import console


def release(label, unload):
    try:
        unload()
    except Exception as error:
        console.detail(f"Could not unload {label}: {error}")


def empty_gpu_cache():
    try:
        import torch
    except ImportError:
        return

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as error:
        console.detail(f"Could not clear the torch VRAM cache: {error}")


def trim_heap():
    try:
        library = ctypes.CDLL("libc.so.6")
    except OSError:
        return

    try:
        library.malloc_trim(0)
    except (AttributeError, OSError):
        pass


def release_everything(reason=""):
    from common import embeddings, language, model, rebel

    release("the language model", model.unload)
    release("the embedding model", embeddings.unload)
    release("REBEL", rebel.unload)
    release("the spaCy pipeline", language.unload)

    gc.collect()
    empty_gpu_cache()
    trim_heap()
    console.detail(f"GPU and memory released{f' ({reason})' if reason else ''}.")

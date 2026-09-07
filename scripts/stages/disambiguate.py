import sys
import time
from pathlib import Path

import numpy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import cleanup, console, embeddings, model, prompts, relations, settings, storage


MERGEABLE_FIELDS = relations.MERGEABLE_FIELDS


def free_the_gpu():
    import gc
    model.unload()
    gc.collect()
    cleanup.empty_gpu_cache()


def means_the_same(first, second):
    answer = relations.ask_for_answer(
        prompts.disambiguation_prompt(first, second), kind="disambiguation"
    )
    if not isinstance(answer, dict):
        return False, None
    return bool(answer.get("same_meaning")), answer


def absorb(kept, candidate, answer):
    merged = {field: answer.get(field) or kept.get(field) for field in MERGEABLE_FIELDS}
    validated = relations.validate(merged, free_type=not settings.current().use_classes)

    if validated is None:
        console.detail(f"The suggested merge '{candidate.get('name')}' -> "
                       f"'{kept.get('name')}' failed validation; keeping both.")
        return False

    kept.update(validated)
    console.detail(f"'{candidate.get('name')}' is redundant with '{kept.get('name')}' — merged.")
    return True


def load_progress(clusters):
    path = settings.DISAMBIGUATE_PROGRESS_FILE
    if not path.exists():
        return {}

    saved = storage.load_json(path, None)
    if saved is None:
        console.detail("The disambiguation checkpoint is unreadable; starting the stage over.")
        return {}

    sizes = {str(cluster_id): len(members) for cluster_id, members in clusters.items()}
    if saved.get("threshold") != settings.SIMILARITY_THRESHOLD or saved.get("sizes") != sizes:
        console.detail("The disambiguation checkpoint belongs to another clustering; "
                       "starting over.")
        return {}

    state = saved.get("clusters", {})
    if not isinstance(state, dict) or not all(
        isinstance(entry, dict) and {"next", "kept", "merged"} <= set(entry)
        for entry in state.values()
    ):
        console.detail("The disambiguation checkpoint is in an old format; starting over.")
        return {}

    return state


def save_progress(clusters, state):
    try:
        storage.save_json_safely({
            "threshold": settings.SIMILARITY_THRESHOLD,
            "sizes": {str(cluster_id): len(members) for cluster_id, members in clusters.items()},
            "clusters": state,
        }, settings.DISAMBIGUATE_PROGRESS_FILE)
    except OSError as error:
        console.detail(f"Could not write the disambiguation checkpoint ({error}); "
                       "the stage goes on.")


def reduce_cluster(members, similarities, saved=None, on_progress=None):
    kept = list(saved["kept"]) if saved else [0]
    start_at = saved["next"] if saved else 1
    merged = dict(saved["merged"]) if saved else {}

    for position, fields in merged.items():
        members[int(position)].update(fields)

    questions = 0
    merges = 0

    for position in range(start_at, len(members)):
        candidate = members[position]
        absorbed = False

        for kept_position in kept:
            if similarities[kept_position, position] < settings.SIMILARITY_THRESHOLD:
                continue

            questions += 1
            same, answer = means_the_same(members[kept_position], candidate)
            if same and answer and absorb(members[kept_position], candidate, answer):
                absorbed = True
                merges += 1
                merged[str(kept_position)] = {
                    field: members[kept_position][field] for field in MERGEABLE_FIELDS
                }
                break

        if not absorbed:
            kept.append(position)

        if on_progress is not None:
            on_progress(position + 1, kept, merged)

    return kept, questions, merges


def question_count(similarities):
    return int(numpy.triu(similarities >= settings.SIMILARITY_THRESHOLD, k=1).sum())


def reduce_clusters(clusters):
    if not clusters:
        return [], []

    free_the_gpu()
    embeddings.load()
    similarities = {
        cluster_id: embeddings.similarity_matrix(members)
        for cluster_id, members in clusters.items()
    }
    embeddings.unload()

    total_before = sum(len(members) for members in clusters.values())
    estimated = sum(question_count(matrix) for matrix in similarities.values())
    console.info(f"{total_before} relation(s) in {len(clusters)} cluster(s); at most {estimated} "
                 f"pair(s) above {settings.SIMILARITY_THRESHOLD:.0%} become a question for the "
                 "model.")

    state = load_progress(clusters)
    started_at = time.time()
    done = 0
    questions_total = 0
    merges_total = 0
    reused = 0
    kept_relations = []

    for position, (cluster_id, members) in enumerate(clusters.items(), start=1):
        key = str(cluster_id)
        saved = state.get(key)

        if saved and saved["next"] >= len(members):
            for place, fields in saved["merged"].items():
                members[int(place)].update(fields)
            kept_relations.extend(members[place] for place in saved["kept"])
            done += len(members)
            reused += 1
            continue

        console.show(
            step="Disambiguating",
            unit=f"cluster {position}/{len(clusters)} ({len(members)} relations)",
            relations=f"{done}/{total_before}",
            elapsed=console.elapsed_since(started_at),
            force=True,
        )

        def on_progress(next_candidate, kept, merged, key=key, position=position, members=members):
            state[key] = {"next": next_candidate, "kept": list(kept), "merged": dict(merged)}
            save_progress(clusters, state)
            console.show(
                unit=f"cluster {position}/{len(clusters)} · {next_candidate}/{len(members)}",
                elapsed=console.elapsed_since(started_at),
            )

        kept, questions, merges = reduce_cluster(
            members, similarities[cluster_id], saved, on_progress
        )
        kept_relations.extend(members[place] for place in kept)
        questions_total += questions
        merges_total += merges
        done += len(members)

    summary = [
        ("Relations in", total_before),
        ("Clusters processed", len(clusters)),
        ("Questions asked", questions_total),
        ("Relations merged away", merges_total),
    ]
    if reused:
        summary.append(("Clusters reused from the checkpoint", reused))

    return kept_relations, summary


def run(options=None):
    options = options or settings.current()

    clusters = storage.load_clusters()
    isolated = storage.load_isolated()
    if not clusters and not isolated:
        console.error(f"No cluster found in {settings.CLUSTERS_DIR} — run the 'cluster' stage "
                      "first.")
        return False

    relations.reset_counts()
    kept, summary = reduce_clusters(clusters)

    if isolated:
        kept.extend(isolated)
        summary.append(("Isolated relations passed through", len(isolated)))

    storage.save_json(kept, settings.DEDUPLICATED_FILE)
    settings.DISAMBIGUATE_PROGRESS_FILE.unlink(missing_ok=True)

    console.clear_status()
    console.success("Disambiguation finished.")
    console.summary([
        *summary,
        ("Relations out", len(kept)),
        *relations.counts_summary(),
        ("Saved to", settings.DEDUPLICATED_FILE.name),
    ])
    return True


if __name__ == "__main__":
    from common import command_line
    sys.exit(command_line.run_single_stage(run, "disambiguate"))

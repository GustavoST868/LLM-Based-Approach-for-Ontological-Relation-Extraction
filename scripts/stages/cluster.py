import gc
import sys
from collections import defaultdict
from pathlib import Path

import numpy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import cleanup, console, embeddings, model, settings, storage


def free_the_gpu():
    model.unload()
    gc.collect()
    cleanup.empty_gpu_cache()


def connected_components(similarities, threshold):
    matrix = numpy.asarray(similarities, dtype=numpy.float32)
    size = matrix.shape[0]
    parents = list(range(size))

    def root_of(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    rows, columns = numpy.nonzero(numpy.triu(matrix >= threshold, k=1))
    for left, right in zip(rows.tolist(), columns.tolist()):
        left_root, right_root = root_of(left), root_of(right)
        if left_root != right_root:
            parents[right_root] = left_root

    numbers = {}
    result = []
    for index in range(size):
        root = root_of(index)
        numbers.setdefault(root, len(numbers))
        result.append(numbers[root])
    return result


def group_relations(found):
    free_the_gpu()
    if not found:
        return {}, []

    embeddings.load()
    console.detail(f"Embedding {len(found)} relations with '{settings.EMBEDDING_MODEL}' "
                   f"(fields: {', '.join(embeddings.COMPARED_FIELDS)}).")
    similarities = embeddings.similarity_matrix(found)

    grouped = defaultdict(list)
    numbers = connected_components(similarities, settings.SIMILARITY_THRESHOLD)
    for relation, cluster_id in zip(found, numbers):
        relation["cluster_id"] = cluster_id
        grouped[cluster_id].append(relation)

    clusters = {}
    isolated = []
    for cluster_id, members in sorted(grouped.items()):
        if len(members) > 1:
            clusters[cluster_id] = members
        else:
            isolated.extend(members)

    embeddings.unload()
    del similarities
    gc.collect()
    free_the_gpu()

    return clusters, isolated


def run(options=None):
    options = options or settings.current()

    if not settings.MERGED_FILE.exists():
        console.error(f"{settings.MERGED_FILE.name} not found — run the 'merge' stage first.")
        return False

    found = storage.load_json(settings.MERGED_FILE, [])
    if not found:
        console.error(f"{settings.MERGED_FILE.name} is empty; there is nothing to group.")
        return False

    storage.clear_clusters()
    settings.DISAMBIGUATE_PROGRESS_FILE.unlink(missing_ok=True)

    console.info(f"Grouping {len(found)} relation(s) by semantic similarity…")
    clusters, isolated = group_relations(found)

    for cluster_id, members in clusters.items():
        storage.save_cluster(cluster_id, members)
    storage.save_isolated(isolated)

    names = {
        f"cluster_{cluster_id}": [relation.get("name") for relation in members if "name" in relation]
        for cluster_id, members in clusters.items()
    }
    names["isolated"] = [relation.get("name") for relation in isolated if "name" in relation]
    storage.save_json(names, settings.CLUSTER_NAMES_FILE)

    largest = max((len(members) for members in clusters.values()), default=0)
    console.success("Clustering finished.")
    console.summary([
        ("Relations in", len(found)),
        ("Similarity threshold", f"{settings.SIMILARITY_THRESHOLD:.0%}"),
        ("Clusters (2+ relations)", len(clusters)),
        ("Largest cluster", largest),
        ("Isolated relations", len(isolated)),
        ("Saved to", f"{settings.CLUSTERS_DIR.name}/ and {settings.CLUSTER_NAMES_FILE.name}"),
    ])
    return True


if __name__ == "__main__":
    from common import command_line
    sys.exit(command_line.run_single_stage(run, "cluster"))

"""Deterministic validation coverage across identities and frame positions."""
from collections import defaultdict


def balanced_validation_indices(people: list[str], variants_per_source: int,
                                count: int) -> list[int]:
    groups = defaultdict(list)
    for index, person in enumerate(people):
        groups[person].append(index)
    names = sorted(groups)
    count = min(count, len(people) * variants_per_source)
    if not names or count < 1:
        return []
    result = []
    for slot, name in enumerate(names):
        quota = count // len(names) + int(slot < count % len(names))
        candidates = [index * variants_per_source + variant
                      for index in groups[name]
                      for variant in range(variants_per_source)]
        quota = min(quota, len(candidates))
        result.append([candidates[min(len(candidates) - 1,
                                     int((k + 0.5) * len(candidates) / quota))]
                       for k in range(quota)])
    return [rows[k] for k in range(max(map(len, result)))
            for rows in result if k < len(rows)]

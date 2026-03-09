from __future__ import annotations

from collections.abc import Sequence
import hashlib
import math
import random
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def select_sections_evenly(
    files: Sequence[T],
    sample_id: str,
    sample_size: int,
    drop_edges: bool = False,
) -> list[T]:
    """
    Deterministically select approximately evenly spaced items for one sample.
    """
    if sample_size <= 0:
        return list(files)

    n = len(files)
    if n == 0:
        return []

    if drop_edges:
        sample_size = sample_size + 2

    if n < sample_size:
        selected = list(files)
        positions = np.arange(len(files))
    else:
        rng = np.random.default_rng(stable_seed(sample_id))
        step = n // (sample_size - 1)
        offset = int(rng.integers(0, step))
        positions = offset + np.arange(sample_size) * step
        positions = positions[positions < n]
        indices = rng.permutation(n)
        selected = [files[int(indices[p])] for p in positions]

    selected = sorted(selected)

    if drop_edges:
        # Preserve prior behavior used in 2_select_representative_sections.py
        if len(positions) - sample_size == 0:
            if len(selected) >= 2:
                selected = selected[1:-1]
        elif len(positions) - sample_size == -1:
            if len(selected) >= 1:
                selected = selected[1:]
        else:
            return []

    return selected


def select_evenly_spaced_items(items: Sequence[T], n_select: int) -> list[T]:
    if n_select <= 0:
        return []
    if n_select >= len(items):
        return list(items)

    spacing = len(items) / n_select
    return [items[math.floor(i * spacing)] for i in range(n_select)]


def greedy_region_coverage_select(
    candidate_ids: Sequence[int],
    regions_by_id: dict[int, set[int]],
    limit: int,
    selected: set[int] | None = None,
    covered_regions: set[int] | None = None,
    secondary_score_by_id: dict[int, tuple[int, ...]] | None = None,
) -> tuple[set[int], set[int]]:
    if selected is None:
        selected = set()
    if covered_regions is None:
        covered_regions = set()
    if secondary_score_by_id is None:
        secondary_score_by_id = {}

    while len(selected) < limit:
        best_id = None
        best_score = None

        for cid in candidate_ids:
            if cid in selected:
                continue
            new_regions = regions_by_id.get(cid, set()) - covered_regions
            score = (len(new_regions),) + secondary_score_by_id.get(cid, ())
            if best_score is None or score > best_score:
                best_score = score
                best_id = cid

        if best_id is None:
            break

        selected.add(best_id)
        covered_regions.update(regions_by_id.get(best_id, set()))

    return selected, covered_regions


def random_fill_selection(
    selected: set[int],
    all_ids: Sequence[int],
    target_total: int,
    rng: random.Random,
) -> set[int]:
    remaining = [cid for cid in all_ids if cid not in selected]
    rng.shuffle(remaining)
    while len(selected) < target_total and remaining:
        selected.add(remaining.pop())
    return selected


def balanced_random_seed_selection(
    group_to_candidate_ids: dict[str, list[int]],
    min_per_group: int,
    target_total: int,
    rng: random.Random,
) -> set[int]:
    selected: set[int] = set()
    for group in sorted(group_to_candidate_ids.keys()):
        candidates = group_to_candidate_ids[group][:]
        rng.shuffle(candidates)
        target_for_group = min(min_per_group, len(candidates))
        for cid in candidates[:target_for_group]:
            if len(selected) >= target_total:
                break
            selected.add(cid)
    return selected

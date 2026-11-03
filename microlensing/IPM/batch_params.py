"""Expansion of a batch description into one fully resolved parameter dict per map."""
import operator
from collections import Counter

import numpy as np

from .ipm import _INPUT_NAMES

SEED_MAX = 2**31 - 1
SEED_BLOCK = 1024


def derive_seeds(start, n):
    """Return n distinct seeds in 1..SEED_MAX from numpy.random.default_rng(start).

    Values are drawn in blocks of exactly SEED_BLOCK and the first occurrence of
    each is kept, so map i's seed does not depend on n."""
    rng = np.random.default_rng(start)
    seeds = []
    seen = set()
    while len(seeds) < n:
        for v in rng.integers(1, SEED_MAX, size=SEED_BLOCK, endpoint=True):
            v = int(v)
            if v not in seen:
                seen.add(v)
                seeds.append(v)
                if len(seeds) == n:
                    break
    return seeds


def _is_sequence(value):
    if isinstance(value, np.ndarray):
        return value.ndim >= 1
    return isinstance(value, (list, tuple, range))


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _scalar(value):
    if isinstance(value, np.ndarray):
        return value.item()
    return _plain(value)


def _seed_start(value):
    """seed_start as an int: an integer (a numpy one too), not a bool, a float or a string, which int() would
    truncate or parse."""
    value = _scalar(value)
    if isinstance(value, bool):
        raise TypeError(f"seed_start must be an integer, got {value!r}")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"seed_start must be an integer, got {value!r}") from None


def _check_seeds(maps):
    for i, m in enumerate(maps):
        if m["starfile"] is not None:
            continue
        v = m["random_seed"]
        if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= SEED_MAX:
            raise ValueError(f"random_seed for map {i} must be an integer in 1..{SEED_MAX}, got {v!r}")


def expand(inputs, num_maps=None, seed_start=None):
    """Return one dict per map, each holding every IPM input name.

    A sequence random_seed gives each map its seed. A single random_seed is one map's seed, so it needs a
    batch with one map that has no starfile. seed_start, an integer, derives one seed per map with derive_seeds,
    and cannot be given together with random_seed. With neither, the seeds are drawn at random."""
    unknown = [k for k in inputs if k not in _INPUT_NAMES]
    if unknown:
        raise TypeError(f"unknown IPM argument(s): {', '.join(map(str, unknown))}")
    if num_maps is not None:
        if isinstance(num_maps, bool) or not isinstance(num_maps, (int, np.integer)) or num_maps < 1:
            raise ValueError(f"num_maps must be a positive integer, got {num_maps!r}")
        num_maps = int(num_maps)

    seqs = {k: [_plain(x) for x in (v.tolist() if isinstance(v, np.ndarray) else list(v))]
            for k, v in inputs.items() if _is_sequence(v)}
    lengths = {k: len(v) for k, v in seqs.items()}
    if any(n == 0 for n in lengths.values()):
        raise ValueError("per-map sequences must not be empty: "
                         + ", ".join(f"{k}: {n}" for k, n in lengths.items()))
    if len(set(lengths.values())) > 1 or (lengths and num_maps is not None
                                          and num_maps != next(iter(lengths.values()))):
        desc = ", ".join(f"{k}: {n}" for k, n in lengths.items())
        extra = f"; num_maps: {num_maps}" if num_maps is not None else ""
        raise ValueError(f"sequence arguments and num_maps disagree in length ({desc}{extra})")
    if lengths:
        n = next(iter(lengths.values()))
    else:
        n = 1 if num_maps is None else num_maps

    shared = {name: _scalar(inputs.get(name)) for name in _INPUT_NAMES if name not in seqs}
    maps = []
    for i in range(n):
        maps.append({name: seqs[name][i] if name in seqs else shared[name] for name in _INPUT_NAMES})

    # seeds
    given = inputs.get("random_seed")
    if given is not None and seed_start is not None:
        raise ValueError("random_seed and seed_start cannot both be given")
    if "random_seed" in seqs:
        _check_seeds(maps)
    elif given is not None:
        seeded = [m for m in maps if m["starfile"] is None]
        if len(seeded) > 1:
            raise ValueError("a single random_seed is one map's seed; give a sequence, or seed_start= to derive "
                             "one seed per map")
        _check_seeds(maps)
    else:
        start = None if seed_start is None else _seed_start(seed_start)
        for m, s in zip(maps, derive_seeds(start, n)):
            m["random_seed"] = s
    for m in maps:
        if m["starfile"] is not None:
            m["random_seed"] = None

    # prefixes
    if "outfile_prefix" not in seqs:
        base = maps[0]["outfile_prefix"] or "./"
        for i, m in enumerate(maps):
            m["outfile_prefix"] = base + f"{i}_"
    prefixes = [m["outfile_prefix"] for m in maps]
    for i, p in enumerate(prefixes):
        if p is None:
            raise ValueError(f"outfile_prefix for map {i} must not be None")
    dups = sorted((p for p, count in Counter(prefixes).items() if count > 1), key=str)
    if dups:
        raise ValueError(f"duplicate outfile_prefix values: {dups}")
    return maps

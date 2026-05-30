"""Data preparation for fine-tuning: MinHash dedup + stratified 70/15/15 split."""

import hashlib
import random
from collections import defaultdict
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ── MinHash deduplication ──────────────────────────────────────────────────────

_DEFAULT_NUM_PERM = 128
_DEFAULT_SHINGLE_SIZE = 3
_DEFAULT_DEDUP_THRESHOLD = 0.8


def _shingles(text: str, k: int = _DEFAULT_SHINGLE_SIZE) -> set[str]:
    """Return word k-shingles from text."""
    words = text.lower().split()
    if len(words) < k:
        return {text.lower()}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def _minhash_signature(text: str, num_perm: int = _DEFAULT_NUM_PERM) -> list[int]:
    """Approximate MinHash signature via universal hashing over shingles."""
    shingled = _shingles(text)
    # Use num_perm independent hash functions simulated by seeding with index
    sig = [2**32] * num_perm
    for shingle in shingled:
        for i in range(num_perm):
            h = int(hashlib.md5(f"{i}|{shingle}".encode(), usedforsecurity=False).hexdigest(), 16)
            h = h % (2**32)
            if h < sig[i]:
                sig[i] = h
    return sig


def _jaccard_estimate(sig1: list[int], sig2: list[int]) -> float:
    matches = sum(a == b for a, b in zip(sig1, sig2, strict=True))
    return matches / len(sig1)


def minhash_dedup(
    texts: list[str],
    threshold: float = _DEFAULT_DEDUP_THRESHOLD,
    num_perm: int = _DEFAULT_NUM_PERM,
) -> list[bool]:
    """Return a boolean keep-mask (True = keep) after MinHash deduplication.

    Uses a greedy O(n²) pass — intended for offline batch preprocessing, not real-time.
    """
    n = len(texts)
    signatures = [_minhash_signature(t, num_perm) for t in texts]
    keep = [True] * n

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            if _jaccard_estimate(signatures[i], signatures[j]) >= threshold:
                keep[j] = False  # mark duplicate as dropped

    dropped = keep.count(False)
    logger.info("minhash_dedup_done", total=n, dropped=dropped, kept=n - dropped)
    return keep


# ── Stratified split ──────────────────────────────────────────────────────────


def stratified_split(
    records: list[dict[str, Any]],
    label_col: str,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into (train, val, test) preserving label distribution.

    For multi-label records, stratification is based on the first label.
    Ratios must sum to 1.0.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"

    rng = random.Random(seed)  # noqa: S311
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for rec in records:
        label = rec.get(label_col)
        key = label[0] if isinstance(label, list) and label else str(label)
        by_label[key].append(rec)

    train, val, test = [], [], []
    for bucket in by_label.values():
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train.extend(bucket[:n_train])
        val.extend(bucket[n_train : n_train + n_val])
        test.extend(bucket[n_train + n_val :])

    rng.shuffle(train)
    logger.info(
        "stratified_split_done",
        train=len(train),
        val=len(val),
        test=len(test),
    )
    return train, val, test


# ── HuggingFace Dataset builders ──────────────────────────────────────────────


def build_icd10_dataset(
    records: list[dict[str, Any]],
    label2id: dict[str, int],
    text_col: str = "raw_text",
) -> Any:
    """Build a HuggingFace Dataset for multi-label ICD-10 classification.

    Each record must have `text_col` and `icd10_codes` (list of str).
    Returns a Dataset with columns: text, labels (multi-hot vector).
    """
    from datasets import Dataset

    num_labels = len(label2id)
    rows = []
    for rec in records:
        text = rec[text_col]
        codes = rec.get("icd10_codes", [])
        multi_hot = [0.0] * num_labels
        for code in codes:
            if code in label2id:
                multi_hot[label2id[code]] = 1.0
        rows.append({"text": text, "labels": multi_hot})

    return Dataset.from_list(rows)


def build_triage_dataset(
    records: list[dict[str, Any]],
    text_col: str = "raw_text",
    label_col: str = "esi_level",
) -> Any:
    """Build a HuggingFace Dataset for 5-class ESI triage classification.

    ESI levels 1-5 are mapped to class indices 0-4.
    """
    from datasets import Dataset

    rows = []
    for rec in records:
        text = rec[text_col]
        esi = int(rec.get(label_col, 3))
        label = max(0, min(4, esi - 1))  # ESI 1–5 → 0–4
        rows.append({"text": text, "labels": label})

    return Dataset.from_list(rows)

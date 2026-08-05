"""Deduplication primitives.

Generic list/id/debug de-duplication used across scoring, routing and the
GraphRetriever class. Kept dependency-free (bottom of the package import graph)
so every other retriever module can import these without a cycle.
"""

from __future__ import annotations

from typing import Any


def _dedupe_debug(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _dedupe_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        normalized = str(value).lower().strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def _dedupe_ids(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        parsed = str(value or "").strip()
        if not parsed:
            continue
        if parsed not in seen:
            seen.add(parsed)
            unique.append(parsed)
    return unique

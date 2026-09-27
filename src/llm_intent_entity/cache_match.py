"""Match cached LLM responses to the items a dataset still needs.

Cache files accumulate entries from earlier datasets. Reconstructing a hit
must not scan the whole evaluation list once per cached row.
"""

from __future__ import annotations

from typing import Any


def evaluation_cache_key(item: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (item["hypothesis"], item["ground_truth"], item.get("context", ""))


def responses_needed_from_cache(
    cached_responses_map: dict[tuple[Any, Any, Any], dict[str, Any]],
    evaluation_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return cached rows whose key is one of `evaluation_items`.

    Order follows the cache map, which is the order the cache file was read.
    """
    needed = {evaluation_cache_key(item) for item in evaluation_items}
    return [
        item
        for cache_key, item in cached_responses_map.items()
        if cache_key in needed
    ]

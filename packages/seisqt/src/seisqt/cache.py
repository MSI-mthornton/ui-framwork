"""Byte-bounded gather cache and neighbour prefetch.

A full-rate gather for 5e3 stations is hundreds of MB, so the cache is bounded
in bytes, not entries. Keys are :meth:`GatherRequest.key` tuples.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Any

log = logging.getLogger(__name__)


class GatherCache:
    """Thread-safe LRU keyed by request, bounded by ``max_bytes``."""

    def __init__(self, max_bytes: int = 1_500_000_000) -> None:
        self.max_bytes = int(max_bytes)
        self._items: OrderedDict[Hashable, tuple[Any, int]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def size_of(value: Any) -> int:
        size = getattr(value, "nbytes", None)
        return int(size) if size is not None else 0

    def get(self, key: Hashable) -> Any | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return item[0]

    def __contains__(self, key: Hashable) -> bool:
        with self._lock:
            return key in self._items

    def put(self, key: Hashable, value: Any) -> None:
        size = self.size_of(value)
        with self._lock:
            if key in self._items:
                self._bytes -= self._items.pop(key)[1]
            if size > self.max_bytes:
                log.info("not caching %s: %d MB exceeds cache size", key, size >> 20)
                return
            self._items[key] = (value, size)
            self._bytes += size
            while self._bytes > self.max_bytes and self._items:
                _, (_, dropped) = self._items.popitem(last=False)
                self._bytes -= dropped

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0

    @property
    def nbytes(self) -> int:
        with self._lock:
            return self._bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class Prefetcher:
    """Warms the cache for events near the current one.

    ``fetch(index)`` must return ``(key, value)`` and is run through the task
    runner at low priority, under a per-slot key so a new selection supersedes
    the previous prefetch round.
    """

    def __init__(
        self,
        runner: Any,
        cache: GatherCache,
        key_for: Callable[[int], Hashable],
        fetch: Callable[[int], Any],
        *,
        ahead: int = 2,
        behind: int = 1,
    ) -> None:
        self.runner = runner
        self.cache = cache
        self.key_for = key_for
        self.fetch = fetch
        self.ahead = ahead
        self.behind = behind
        self.enabled = True

    def around(self, index: int, n_events: int) -> list[int]:
        if not self.enabled or n_events <= 0:
            return []
        wanted = [index + d for d in range(1, self.ahead + 1)]
        wanted += [index - d for d in range(1, self.behind + 1)]
        todo = [i for i in wanted if 0 <= i < n_events and self.key_for(i) not in self.cache]
        for slot, i in enumerate(todo):
            self.runner.submit(
                lambda i=i: self.fetch(i),
                self._store,
                key=f"prefetch:{slot}",
                label=f"prefetch event {i}",
                priority=-1,
            )
        return todo

    def _store(self, result: Any) -> None:
        if result.ok and result.value is not None:
            key, value = result.value
            self.cache.put(key, value)

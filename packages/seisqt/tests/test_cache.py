import numpy as np

from seisqt.cache import GatherCache


class Blob:
    def __init__(self, n):
        self.nbytes = n


def test_lru_bounded_in_bytes():
    cache = GatherCache(max_bytes=100)
    cache.put("a", Blob(40))
    cache.put("b", Blob(40))
    assert cache.get("a") is not None  # a is now most recent
    cache.put("c", Blob(40))
    assert "b" not in cache and "a" in cache and "c" in cache
    assert cache.nbytes == 80
    cache.put("huge", Blob(1000))
    assert "huge" not in cache
    cache.put("arr", np.zeros(5))
    assert cache.nbytes <= 100

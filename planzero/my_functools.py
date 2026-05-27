import functools
import os
import shutil

try:
    import diskcache
except ImportError:
    diskcache = None

CACHE_DIR = os.environ['PLANZERO_CACHE_DIR']
USE_DISK_CACHE = (os.environ['PLANZERO_USE_DISK_CACHE'] == '1')

_disk_cache = None

def cache(f):
    """Typically used as a decorator.

    This decorator always acts as a cache.
    It always acts as a memory cache.
    It sometimes also acts as a disk cache.

    If a function is only called via this decorator, then it will be executed
    either zero or one times per process lifetime.
    """
    if USE_DISK_CACHE and diskcache:
        global _disk_cache
        if _disk_cache is None:
            _disk_cache = diskcache.Cache(CACHE_DIR)
        return functools.cache(_disk_cache.memoize()(f))
    else:
        return functools.cache(f)

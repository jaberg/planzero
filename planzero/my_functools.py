import functools
import os
import shutil

try:
    import diskcache
except ImportError:
    diskcache = None

CACHE_DIR = os.environ['PLANZERO_CACHE_DIR']
USE_DISK_CACHE = os.environ['PLANZERO_USE_DISK_CACHE'] == '1'

_disk_cache = None

def cache(f):
    if USE_DISK_CACHE and diskcache:
        global _disk_cache
        if _disk_cache is None:
            _disk_cache = diskcache.Cache(CACHE_DIR)
        return _disk_cache.memoize()(f)
    else:
        return functools.cache(f)

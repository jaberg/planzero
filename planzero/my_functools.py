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

_inference_cache = None

def inference_cache(recompute=False):
    """
    A decorator that caches function results to disk.

    Args:
        recompute (bool): If True, bypasses the cache, forces the function
                          to execute, and updates the cache with the new result.
                          If False, uses standard caching behavior.
    """
    global _inference_cache
    if _inference_cache is None:
        _inference_cache = diskcache.Cache('./inference_cache/')

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a deterministic key for the cache based on function and arguments
            cache_key = (
                func.__module__,
                func.__name__,
                args,
                frozenset(kwargs.items())
            )

            # If recompute is True, skip checking the cache and force execution
            if recompute:
                result = func(*args, **kwargs)
                _inference_cache[cache_key] = result
                return result

            # Default behavior: Return cached value if it exists
            if cache_key in _inference_cache:
                return _inference_cache[cache_key]

            # Otherwise, compute the value and store it in the cache
            result = func(*args, **kwargs)
            _inference_cache[cache_key] = result
            return result

        return wrapper
    return decorator

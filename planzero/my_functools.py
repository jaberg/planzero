import functools

def maybecache(f):
    if 1:
        return f
    else:
        return functools.cache(f)

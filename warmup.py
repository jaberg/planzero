import time

from fastapi.testclient import TestClient

import app
import planzero.endpoints

populate_cache = 0

if app._app_cache is None:
    # yes, not actually cached, it's okay.
    # We're probably in a development environment and calling
    # the endpoints function is fine because all of the dependencies
    # are installed.
    def cached_endpoints():
        return list(planzero.endpoints.endpoints())
else:
    # This cache prevents calling endpoints in production
    # because endpoints loads files that won't work in prod.
    #
    # called in dev environment by test_200.py test_endpoints
    # name is specified so that the cache key is the same when this
    # file is __main__ and if it is a module.
    @app._app_cache.memoize(name='warmup.cached_endpoints')
    def cached_endpoints():
        return list(planzero.endpoints.endpoints())

def warmup():
    client = TestClient(app.app)
    if populate_cache:
        # populate the disk cache
        for endpoint in cached_endpoints():
            response = client.get(endpoint)
            assert response.status_code == 200
            print(response.status_code, endpoint)

    # time the accesses, they should be quick
    times = []
    for endpoint in cached_endpoints():
        t0 = time.time()
        response = client.get(endpoint)
        t1 = time.time()
        dt = t1 - t0
        print(response.status_code, f'{dt:.2f}', endpoint)
        times.append((dt, endpoint))
    times.sort()
    print('------------------------')
    print('Top 10 Slowest Endpoints')
    print('------------------------')
    for dt, endpoint in reversed(times[-10:]):
        print(dt, endpoint)

if __name__ == "__main__":
    warmup()

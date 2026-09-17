import time

from fastapi.testclient import TestClient

import app
import planzero.endpoints

populate_cache = 0

# called in dev environment by test_200.py test_endpoints
@app.app_cache
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

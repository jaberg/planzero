import os
import shutil
import sys
import time

# Ensure the planzero package can be imported
sys.path.append(os.getcwd())

# Enable disk caching
os.environ['PLANZERO_USE_DISK_CACHE'] = '1'

CACHE_DIR = '.planzero_cache'
os.environ['PLANZERO_CACHE_DIR'] = CACHE_DIR

if os.path.exists(CACHE_DIR):
    shutil.rmtree(CACHE_DIR)

from fastapi.testclient import TestClient

import app
import planzero

def warmup():
    client = TestClient(app.app)
    # populate the disk cache
    for endpoint in planzero.endpoints.endpoints():
        response = client.get(endpoint)
        assert response.status_code == 200
        print(response.status_code, endpoint)

    # time the accesses, they should be quick
    times = []
    for endpoint in planzero.endpoints.endpoints():
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

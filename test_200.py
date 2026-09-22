import re
import time
import warnings
from urllib.parse import urljoin

import pytest
from fastapi.testclient import TestClient

import app
import warmup

client = TestClient(app.app)


_checked_links = set()

@pytest.mark.parametrize("endpoint", warmup.cached_endpoints())
def test_endpoints(
        endpoint,
        page_build_duration_limit=20.0, # seconds
        ):
    t0 = time.time()
    response = client.get(endpoint)
    assert response.status_code == 200
    _checked_links.add(endpoint)
    t1 = time.time()
    duration = t1 - t0
    if duration > page_build_duration_limit:
        warnings.warn(f"endpoint {endpoint} took {duration} seconds to render")


@pytest.mark.parametrize("endpoint", warmup.cached_endpoints())
def test_internal_links(endpoint):
    response = client.get(endpoint)
    assert response.status_code == 200

    # Extract all href attributes, excluding fragments and query parameters
    html = response.text
    links = re.findall(r'href=["\']([^"\'#?]+)["\']', html)
    assert 'StrictUndefined' not in html

    for link in set(links):
        # Skip external protocols
        if re.match(r'^(https?:|mailto:|tel:|ftp:)', link):
            continue

        # Resolve the link relative to the current endpoint
        # urljoin correctly handles absolute paths starting with "/"
        # and relative paths without a leading "/"
        absolute_link = urljoin(endpoint, link)

        if absolute_link not in _checked_links:
            res = client.get(absolute_link)
            assert res.status_code == 200, f"Link '{link}' (resolved to '{absolute_link}') on page {endpoint} is broken"
            _checked_links.add(absolute_link)


def test_blog_404():
    url = "/blog/not-an-actual-blog/"
    response = client.get(url)
    assert response.status_code == 404

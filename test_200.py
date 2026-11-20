from html.parser import HTMLParser
from urllib.parse import unquote, urljoin, urlsplit

import pytest
from fastapi.testclient import TestClient

import app
import warmup

client = TestClient(app.app)


_checked_links = set()
_parsed_pages = {}


class _PageParser(HTMLParser):
    """Collect the links and element ids from a rendered page."""

    def __init__(self):
        super().__init__()
        self.hrefs = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        href = attributes.get('href')
        if href is not None:
            self.hrefs.append(href)
        id_ = attributes.get('id')
        if id_ is not None:
            self.ids.add(id_)


def _page_url(url):
    """Return a URL without its query string or fragment."""
    return urlsplit(url)._replace(query='', fragment='').geturl()


def _parse_response(response):
    parser = _PageParser()
    parser.feed(response.text)
    return parser


def _get_page(path, *, link=None, endpoint=None):
    """Fetch and parse a page once, reusing the cached parse thereafter."""
    if path not in _parsed_pages:
        res = client.get(path)
        assert res.status_code == 200, f"Link '{link}' (resolved to '{path}') on page {endpoint} is broken"
        _parsed_pages[path] = _parse_response(res)
        _checked_links.add(path)
    return _parsed_pages[path]


@pytest.mark.parametrize("endpoint", warmup.cached_endpoints())
def test_endpoints(
        endpoint,
        ):
    response = client.get(endpoint)
    assert response.status_code == 200
    _checked_links.add(endpoint)


@pytest.mark.parametrize("endpoint", warmup.cached_endpoints())
def test_internal_links(endpoint):
    response = client.get(endpoint)
    assert response.status_code == 200

    html = response.text
    parser = _parse_response(response)
    assert 'StrictUndefined' not in html
    assert 'TODO' not in html
    assert 'XXX' not in html

    _checked_links.add(endpoint)
    _parsed_pages[endpoint] = parser

    current_path = _page_url(endpoint)

    for link in set(parser.hrefs):
        link = link.strip()
        split_link = urlsplit(link)

        # Skip external protocols and protocol-relative URLs
        if split_link.scheme or split_link.netloc:
            continue

        absolute_link = urljoin(endpoint, link)
        target = _page_url(absolute_link) or current_path
        fragment = unquote(urlsplit(absolute_link).fragment)

        if target == current_path:
            ids = parser.ids
        else:
            ids = _get_page(target, link=link, endpoint=endpoint).ids

        if fragment:
            assert fragment in ids, (
                f"Fragment '#{fragment}' in link '{link}' "
                f"(resolved to '{target}#{fragment}') on page {endpoint} "
                f"has no matching id"
            )


def test_blog_404():
    url = "/blog/not-an-actual-blog/"
    response = client.get(url)
    assert response.status_code == 404

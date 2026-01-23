import pytest
from fastapi.testclient import TestClient

import app
peval = app.get_peval()

client = TestClient(app.app)

from planzero import ipcc_canada
import planzero.blog

def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_home_ipcc_sectors():
    response = client.get("/ipcc-sectors/")
    assert response.status_code == 200


def test_home_strategies():
    response = client.get("/strategies/")
    assert response.status_code == 200


def test_home_about():
    response = client.get("/about/")
    assert response.status_code == 200


@pytest.mark.parametrize("catpath", ipcc_canada.catpaths)
def test_each_ipcc_sector(catpath):
    assert len(ipcc_canada.catpaths) == 71 # what should this number be?
    url = f"/ipcc-sectors/{catpath}/"
    response = client.get(url)
    assert response.status_code == 200

@pytest.mark.parametrize("idea_name", peval.projects)
def test_each_strategy(idea_name):
    assert len(peval.projects) > 3
    url = f"/strategies/{idea_name}/"
    response = client.get(url)
    assert response.status_code == 200


@pytest.mark.parametrize("url_filename", planzero.blog._blogs_by_url_filename)
def test_each_blog(url_filename):
    assert len(planzero.blog._blogs_by_url_filename) == 1
    url = f"/blog/{url_filename}/"
    response = client.get(url)
    assert response.status_code == 200

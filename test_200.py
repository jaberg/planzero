import pytest
from fastapi.testclient import TestClient

import app

client = TestClient(app.app)

import ipcc_canada

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

@pytest.mark.parametrize("idea_name", app.peval.projects)
def test_each_strategy(idea_name):
    assert len(app.peval.projects) > 3
    url = f"/strategies/{idea_name}/"
    response = client.get(url)
    assert response.status_code == 200


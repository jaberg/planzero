import pytest
from fastapi.testclient import TestClient
import app
import planzero

client = TestClient(app.app)


@pytest.mark.parametrize("endpoint", planzero.endpoints.endpoints())
def test_endpoints(endpoint):
    response = client.get(endpoint)
    assert response.status_code == 200


def test_blog_404():
    url = f"/blog/not-an-actual-blog/"
    response = client.get(url)
    assert response.status_code == 404

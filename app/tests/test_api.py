
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import init_db

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    # Initialize DB before tests
    init_db()
    yield

def test_metadata_empty():
    response = client.get("/metadata")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
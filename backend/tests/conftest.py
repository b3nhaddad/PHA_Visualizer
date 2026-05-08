"""Shared pytest fixtures."""
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def pure_request() -> dict:
    return {"model": {"type": "pure_p_spin", "p": 3, "N": 10, "seed": 1}, "n_steps": 20}


@pytest.fixture
def mixed_request() -> dict:
    return {
        "model": {
            "type": "mixed_p_spin",
            "N": 10,
            "components": [{"p": 2, "beta": 0.6}, {"p": 3, "beta": 0.8}],
            "seed": 7,
        },
        "n_steps": 20,
    }

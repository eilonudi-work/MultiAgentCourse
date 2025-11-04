"""Tests for health check endpoints."""
import pytest
from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    """Test basic health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "version" in data
    assert "checks" in data
    assert "database" in data["checks"]


def test_metrics_endpoint(client: TestClient):
    """Test metrics endpoint."""
    # Make a few requests to generate metrics
    client.get("/health")
    client.get("/")

    response = client.get("/metrics")
    assert response.status_code == 200

    data = response.json()
    assert "enabled" in data
    if data["enabled"]:
        assert "summary" in data
        assert "endpoints" in data


def test_metrics_summary(client: TestClient):
    """Test metrics summary endpoint."""
    response = client.get("/metrics/summary")
    assert response.status_code == 200

    data = response.json()
    assert "enabled" in data


def test_backup_status(client: TestClient):
    """Test backup status endpoint."""
    response = client.get("/backup/status")
    assert response.status_code == 200

    data = response.json()
    assert "enabled" in data


def test_info_endpoint(client: TestClient):
    """Test API info endpoint."""
    response = client.get("/info")
    assert response.status_code == 200

    data = response.json()
    assert data["name"] == "Ollama Web GUI API"
    assert "version" in data
    assert "features" in data
    assert "configuration" in data


def test_root_endpoint(client: TestClient):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["name"] == "Ollama Web GUI API"
    assert "version" in data
    assert "docs" in data
    assert "health" in data

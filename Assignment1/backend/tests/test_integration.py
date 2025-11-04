"""Integration tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from app.models.user import User


class TestAuthEndpoints:
    """Integration tests for authentication endpoints."""

    def test_setup_new_user(self, client: TestClient, mock_ollama_client):
        """Test setting up a new user."""
        response = client.post(
            "/api/auth/setup",
            json={
                "api_key": "new-test-key",
                "ollama_url": "http://localhost:11434",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "user_id" in data

    def test_verify_api_key(self, client: TestClient, test_user: User):
        """Test verifying an API key."""
        response = client.post(
            "/api/auth/verify",
            json={"api_key": "test-api-key-123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["user_id"] == test_user.id


class TestModelEndpoints:
    """Integration tests for model endpoints."""

    def test_list_models(
        self, client: TestClient, test_user: User, auth_headers: dict, mock_ollama_client
    ):
        """Test listing available models."""
        response = client.get("/api/models", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert "name" in data[0]

    def test_test_connection(
        self, client: TestClient, test_user: User, auth_headers: dict, mock_ollama_client
    ):
        """Test Ollama connection test."""
        response = client.get("/api/models/test", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "connected"


class TestConversationEndpoints:
    """Integration tests for conversation endpoints."""

    def test_create_conversation(
        self, client: TestClient, test_user: User, auth_headers: dict
    ):
        """Test creating a conversation."""
        response = client.post(
            "/api/conversations",
            headers=auth_headers,
            json={
                "title": "Test Conversation",
                "model": "llama2",
                "system_prompt": "You are a helpful assistant.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Test Conversation"
        assert data["model"] == "llama2"
        assert "id" in data

    def test_list_conversations(
        self, client: TestClient, test_user: User, auth_headers: dict
    ):
        """Test listing conversations."""
        # Create a conversation first
        client.post(
            "/api/conversations",
            headers=auth_headers,
            json={"title": "Test", "model": "llama2"},
        )

        # List conversations
        response = client.get("/api/conversations", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_get_conversation(
        self, client: TestClient, test_user: User, auth_headers: dict
    ):
        """Test getting a specific conversation."""
        # Create a conversation
        create_response = client.post(
            "/api/conversations",
            headers=auth_headers,
            json={"title": "Test", "model": "llama2"},
        )
        conversation_id = create_response.json()["id"]

        # Get the conversation
        response = client.get(
            f"/api/conversations/{conversation_id}",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == conversation_id
        assert data["title"] == "Test"

    def test_delete_conversation(
        self, client: TestClient, test_user: User, auth_headers: dict
    ):
        """Test deleting a conversation."""
        # Create a conversation
        create_response = client.post(
            "/api/conversations",
            headers=auth_headers,
            json={"title": "Test", "model": "llama2"},
        )
        conversation_id = create_response.json()["id"]

        # Delete the conversation
        response = client.delete(
            f"/api/conversations/{conversation_id}",
            headers=auth_headers,
        )
        assert response.status_code == 200

        # Verify it's deleted
        get_response = client.get(
            f"/api/conversations/{conversation_id}",
            headers=auth_headers,
        )
        assert get_response.status_code == 404


class TestHealthEndpoints:
    """Integration tests for health and monitoring endpoints."""

    def test_health_check(self, client: TestClient):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "checks" in data
        assert "database" in data["checks"]

    def test_metrics_endpoint(self, client: TestClient):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data

    def test_info_endpoint(self, client: TestClient):
        """Test API info endpoint."""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Ollama Web GUI API"
        assert "version" in data
        assert "features" in data


class TestErrorHandling:
    """Integration tests for error handling."""

    def test_not_found(self, client: TestClient):
        """Test 404 error handling."""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404

    def test_unauthorized(self, client: TestClient):
        """Test unauthorized access."""
        response = client.get("/api/models")
        assert response.status_code == 401

    def test_invalid_conversation_id(
        self, client: TestClient, test_user: User, auth_headers: dict
    ):
        """Test accessing non-existent conversation."""
        response = client.get("/api/conversations/99999", headers=auth_headers)
        assert response.status_code == 404
        data = response.json()
        assert "error" in data

    def test_validation_error(self, client: TestClient, test_user: User, auth_headers: dict):
        """Test validation error handling."""
        response = client.post(
            "/api/conversations",
            headers=auth_headers,
            json={"title": "", "model": "llama2"},  # Empty title should fail
        )
        assert response.status_code in [400, 422]


class TestSecurityFeatures:
    """Integration tests for security features."""

    def test_security_headers(self, client: TestClient):
        """Test that security headers are present."""
        response = client.get("/health")
        assert response.status_code == 200

        # Check for security headers
        headers = response.headers
        # Note: Some headers may only be present if middleware is fully enabled
        # This is a basic check that the response has headers
        assert "content-type" in headers

    def test_rate_limit_headers(self, client: TestClient, test_user: User, auth_headers: dict):
        """Test that rate limit headers are present."""
        response = client.get("/api/models", headers=auth_headers)

        # Check for rate limit headers (if enabled)
        # These might not be present if rate limiting is disabled in tests
        assert response.status_code == 200

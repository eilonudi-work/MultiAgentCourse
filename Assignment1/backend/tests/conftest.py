"""Pytest configuration and fixtures."""
import pytest
import os
from typing import Generator
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.main import app
from app.database import Base, get_db
from app.models.user import User
from app.utils.auth import hash_api_key

# Use in-memory SQLite for tests
TEST_DATABASE_URL = "sqlite:///:memory:"

# Create test engine
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
)

# Create test session
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="function")
def db() -> Generator[Session, None, None]:
    """
    Create a fresh database for each test.

    Yields:
        Database session
    """
    # Create tables
    Base.metadata.create_all(bind=test_engine)

    # Create session
    session = TestingSessionLocal()

    try:
        yield session
    finally:
        session.close()
        # Drop all tables
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def client(db: Session) -> TestClient:
    """
    Create a test client with database override.

    Args:
        db: Database session fixture

    Returns:
        FastAPI test client
    """
    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def test_user(db: Session) -> User:
    """
    Create a test user.

    Args:
        db: Database session

    Returns:
        Test user
    """
    user = User(
        api_key_hash=hash_api_key("test-api-key-123"),
        ollama_url="http://localhost:11434",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def auth_headers() -> dict:
    """
    Get authorization headers for test user.

    Returns:
        Headers dictionary
    """
    return {"Authorization": "Bearer test-api-key-123"}


@pytest.fixture
def mock_ollama_client(monkeypatch):
    """
    Mock Ollama client for testing.

    Args:
        monkeypatch: Pytest monkeypatch fixture
    """
    class MockOllamaClient:
        def __init__(self, base_url=None):
            self.base_url = base_url

        async def test_connection(self):
            return True

        async def list_models(self, retry_count=3):
            return [
                {
                    "name": "llama2",
                    "model": "llama2",
                    "size": 3825819519,
                    "modified_at": "2024-01-01T00:00:00Z",
                },
                {
                    "name": "codellama",
                    "model": "codellama",
                    "size": 3825819519,
                    "modified_at": "2024-01-01T00:00:00Z",
                },
            ]

        async def get_model_info(self, model_name):
            if model_name in ["llama2", "codellama"]:
                return {
                    "name": model_name,
                    "model": model_name,
                    "size": 3825819519,
                }
            return None

        async def is_ollama_available(self):
            return True

        async def stream_chat(self, model, messages, temperature=0.7):
            # Mock streaming response
            yield {"message": {"content": "Hello"}, "done": False}
            yield {"message": {"content": " "}, "done": False}
            yield {"message": {"content": "world"}, "done": False}
            yield {"message": {"content": "!"}, "done": True}

        async def close(self):
            pass

    # Patch the get_ollama_client function
    from app.services import ollama_client
    monkeypatch.setattr(ollama_client, "get_ollama_client", lambda base_url=None: MockOllamaClient(base_url))

    return MockOllamaClient


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset metrics before each test."""
    from app.utils.metrics import get_metrics_collector
    collector = get_metrics_collector()
    collector.reset()
    yield
    collector.reset()


@pytest.fixture(autouse=True)
def disable_rate_limiting(monkeypatch):
    """Disable rate limiting for tests."""
    from app import config
    monkeypatch.setattr(config.settings, "RATE_LIMIT_ENABLED", False)


@pytest.fixture(autouse=True)
def disable_csrf(monkeypatch):
    """Disable CSRF protection for tests."""
    from app import config
    monkeypatch.setattr(config.settings, "CSRF_PROTECTION_ENABLED", False)

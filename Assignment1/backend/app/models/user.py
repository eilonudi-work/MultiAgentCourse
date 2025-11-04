"""User model for authentication and configuration."""
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import relationship
from app.database import Base


class User(Base):
    """User model for storing API key hashes and Ollama configurations."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    api_key_hash = Column(String, unique=True, nullable=False, index=True)
    ollama_url = Column(String, default="http://localhost:11434", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Session management fields
    last_activity = Column(DateTime, default=datetime.utcnow, nullable=False)
    session_expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)

    # API key management
    api_key_created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    api_key_expires_at = Column(DateTime, nullable=True)

    # Admin flag
    is_admin = Column(Boolean, default=False, nullable=False)

    # Relationships
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    settings = relationship("Setting", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, ollama_url={self.ollama_url})>"

    def is_session_valid(self) -> bool:
        """
        Check if user session is still valid.

        Returns:
            True if session is valid, False otherwise
        """
        if not self.is_active:
            return False

        if self.session_expires_at and datetime.utcnow() > self.session_expires_at:
            return False

        return True

    def is_api_key_valid(self) -> bool:
        """
        Check if API key is still valid.

        Returns:
            True if API key is valid, False otherwise
        """
        if self.api_key_expires_at and datetime.utcnow() > self.api_key_expires_at:
            return False

        return True

    def update_activity(self, session_timeout_minutes: int = 60) -> None:
        """
        Update last activity timestamp and session expiration.

        Args:
            session_timeout_minutes: Session timeout in minutes
        """
        self.last_activity = datetime.utcnow()
        self.session_expires_at = datetime.utcnow() + timedelta(minutes=session_timeout_minutes)

    def revoke_session(self) -> None:
        """Revoke user session."""
        self.session_expires_at = datetime.utcnow()
        self.is_active = False

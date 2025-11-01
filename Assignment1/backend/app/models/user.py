"""User model for authentication and configuration."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from app.database import Base


class User(Base):
    """User model for storing API key hashes and Ollama configurations."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    api_key_hash = Column(String, unique=True, nullable=False, index=True)
    ollama_url = Column(String, default="http://localhost:11434", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    settings = relationship("Setting", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, ollama_url={self.ollama_url})>"

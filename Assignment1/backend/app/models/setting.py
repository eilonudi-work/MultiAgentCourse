"""Setting model for user preferences."""
from sqlalchemy import Column, Integer, String, Text, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from app.database import Base


class Setting(Base):
    """Setting model for storing user-specific configuration."""

    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    key = Column(String, nullable=False)
    value = Column(Text, nullable=True)

    # Ensure unique key per user
    __table_args__ = (
        UniqueConstraint("user_id", "key", name="unique_user_setting"),
    )

    # Relationships
    user = relationship("User", back_populates="settings")

    def __repr__(self):
        return f"<Setting(user_id={self.user_id}, key={self.key})>"

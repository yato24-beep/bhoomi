from datetime import datetime, timezone
from enum import Enum
from sqlalchemy import Integer, String, Boolean, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class UserRole(str, Enum):
    """User authorization roles."""
    ADMIN = "ADMIN"        # Full system access (upload, view, edit, delete, user management)
    OFFICER = "OFFICER"    # Ingestion operator (upload & process documents, view results)
    REVIEWER = "REVIEWER"  # Verification analyst (view documents, review extraction accuracy)
    VIEWER = "VIEWER"      # Read-only stakeholder (view documents & results)


class User(Base):
    """User account entity with role-based access control."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=True)
    role: Mapped[str] = mapped_column(
        String(50),
        default=UserRole.VIEWER.value,
        nullable=False,
        index=True,
        doc="ADMIN, OFFICER, REVIEWER, or VIEWER",
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email='{self.email}', role='{self.role}')>"

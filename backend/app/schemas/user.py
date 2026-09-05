from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class UserBase(BaseModel):
    """Base user schema."""
    email: str = Field(..., max_length=255, description="Unique user email address")
    full_name: Optional[str] = Field(default=None, max_length=255, description="Full display name")
    role: str = Field(default="VIEWER", description="Role: ADMIN, OFFICER, REVIEWER, or VIEWER")


class UserCreate(UserBase):
    """Schema for registering a new user."""
    password: str = Field(..., min_length=6, max_length=128, description="Plain text password")


class UserRead(UserBase):
    """Schema for returning user data."""
    id: int = Field(..., description="Unique user database ID")
    is_active: bool = Field(..., description="Account active status")
    created_at: datetime = Field(..., description="Account creation timestamp")

    model_config = ConfigDict(from_attributes=True)

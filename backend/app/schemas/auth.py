from pydantic import BaseModel, Field
from app.schemas.user import UserRead


class Token(BaseModel):
    """Authentication response payload containing JWT token and user info."""
    access_token: str = Field(..., description="JWT Bearer access token")
    token_type: str = Field(default="bearer", description="Token type")
    user: UserRead = Field(..., description="Logged-in user profile")


class LoginRequest(BaseModel):
    """Schema for user login credentials."""
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password")

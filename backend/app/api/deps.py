from typing import Generator, List, Callable, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.config import settings
from app.db.session import get_db
from app.core.security import decode_access_token
from app.models.user import User, UserRole

# OAuth2 scheme for FastAPI Swagger UI authorization
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login-form",
    auto_error=False,
)


def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Extract and validate JWT token to return User, or None if unauthenticated."""
    if not token:
        return None

    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        return None

    user_id_str = payload.get("sub")
    try:
        user_id = int(user_id_str)
    except (ValueError, TypeError):
        return None

    stmt = select(User).where(User.id == user_id)
    user = db.execute(stmt).scalar_one_or_none()
    return user


def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Strictly require authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials. Please provide a valid JWT Bearer token.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token:
        raise credentials_exception

    user = get_current_user_optional(token, db)
    if not user:
        raise credentials_exception
    return user


def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Verify that the authenticated user account is active."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user account",
        )
    return current_user


def require_roles(allowed_roles: List[str]) -> Callable:
    """Dependency factory enforcing Role-Based Access Control (RBAC)."""
    def role_checker(
        current_user: Optional[User] = Depends(get_current_user_optional),
    ) -> Optional[User]:
        # If user provided a token, strictly enforce role
        if current_user:
            if current_user.role not in allowed_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=(
                        f"Permission denied: role '{current_user.role}' is not authorized. "
                        f"Required roles: {', '.join(allowed_roles)}"
                    ),
                )
            return current_user
        
        # If no token provided:
        # If ADMIN role required, strictly demand authentication
        if UserRole.ADMIN.value in allowed_roles and len(allowed_roles) == 1:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required for administrative actions.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # For general access, allow guest operation
        return None

    return role_checker


# Predefined RBAC dependencies
require_admin = require_roles([UserRole.ADMIN.value])
require_officer_or_admin = require_roles([UserRole.ADMIN.value, UserRole.OFFICER.value])
require_reviewer_or_above = require_roles([UserRole.ADMIN.value, UserRole.OFFICER.value, UserRole.REVIEWER.value])
require_viewer_or_above = require_roles([
    UserRole.ADMIN.value,
    UserRole.OFFICER.value,
    UserRole.REVIEWER.value,
    UserRole.VIEWER.value,
])

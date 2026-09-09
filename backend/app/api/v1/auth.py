from fastapi import APIRouter, Depends, Form, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.api.deps import get_db, get_current_active_user
from app.core.security import get_password_hash, verify_password, create_access_token
from app.models.user import User, UserRole
from app.schemas.user import UserCreate, UserRead
from app.schemas.auth import Token, LoginRequest

router = APIRouter()


@router.post(
    "/register",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    summary="Register New User",
    description="Create a new user account with assigned role (ADMIN, OFFICER, REVIEWER, VIEWER).",
)
def register_user(
    user_in: UserCreate,
    db: Session = Depends(get_db),
) -> User:
    """Register a new user account."""
    # Check if user with this email already exists
    stmt = select(User).where(User.email == user_in.email.lower().strip())
    existing_user = db.execute(stmt).scalar_one_or_none()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User with email '{user_in.email}' already exists",
        )

    # Validate role
    role_val = user_in.role.upper()
    valid_roles = [r.value for r in UserRole]
    if role_val not in valid_roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role '{user_in.role}'. Allowed roles: {', '.join(valid_roles)}",
        )

    # Create user record
    hashed_password = get_password_hash(user_in.password)
    user = User(
        email=user_in.email.lower().strip(),
        hashed_password=hashed_password,
        full_name=user_in.full_name,
        role=role_val,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post(
    "/login",
    response_model=Token,
    summary="User Login (JSON)",
    description="Authenticate user with email and password, returning JWT access token.",
)
def login_json(
    login_data: LoginRequest,
    db: Session = Depends(get_db),
) -> Token:
    """Authenticate with JSON payload and return JWT token."""
    stmt = select(User).where(User.email == login_data.email.lower().strip())
    user = db.execute(stmt).scalar_one_or_none()

    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    access_token = create_access_token(
        subject=user.id,
        role=user.role,
        email=user.email,
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserRead.model_validate(user),
    )


@router.post(
    "/login-form",
    summary="Swagger OAuth2 Login Form",
    description="OAuth2 password form endpoint used by Swagger UI Authorize modal.",
)
def login_oauth2_form(
    username: str = Form(..., description="User email / username"),
    password: str = Form(..., description="User password"),
    grant_type: str = Form(default="password"),
    db: Session = Depends(get_db),
):
    """OAuth2 compatible token login for Swagger UI."""
    stmt = select(User).where(User.email == username.lower().strip())
    user = db.execute(stmt).scalar_one_or_none()

    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        subject=user.id,
        role=user.role,
        email=user.email,
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
    }


@router.get(
    "/me",
    response_model=UserRead,
    summary="Get Current User Profile",
    description="Fetch details of the currently authenticated user from JWT token.",
)
def get_current_user_profile(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """Return currently logged-in user profile."""
    return current_user

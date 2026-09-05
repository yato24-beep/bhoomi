from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
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

    new_user = User(
        email=user_in.email.lower().strip(),
        hashed_password=get_password_hash(user_in.password),
        full_name=user_in.full_name,
        role=role_val,
        is_active=True,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@router.post(
    "/login",
    response_model=Token,
    summary="User Login (JSON)",
    description="Authenticate with email and password to receive a JWT access token.",
)
def login_json(
    credentials: LoginRequest,
    db: Session = Depends(get_db),
) -> Token:
    """Authenticate user and return JWT Bearer token."""
    stmt = select(User).where(User.email == credentials.email.lower().strip())
    user = db.execute(stmt).scalar_one_or_none()

    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User account is deactivated",
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
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """OAuth2 compatible token login for Swagger UI."""
    stmt = select(User).where(User.email == form_data.username.lower().strip())
    user = db.execute(stmt).scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.hashed_password):
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

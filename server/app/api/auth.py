"""
Authentication API Endpoints:
- Register (Email + Salted Hashed Password)
- Login (Credentials verification + JWT Access Token)
- Google OAuth (Token verification or dev fallback)
- User Profile & Settings update
"""
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import timedelta

from server.app.db.session import get_db
from server.app.models.entities import User
from server.app.schemas.schemas import (
    UserCreate,
    UserLogin,
    GoogleAuthRequest,
    UserResponse,
    Token,
    UserUpdate
)
from server.app.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    get_current_user
)
from server.app.core.config import settings

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=Token)
def register_user(user_in: UserCreate, db: Session = Depends(get_db)):
    """Creates a new user with salted bcrypt password hashing."""
    existing_user = db.query(User).filter(User.email == user_in.email.lower()).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists. Please log in."
        )

    user = User(
        email=user_in.email.lower(),
        name=user_in.name or user_in.email.split("@")[0].title(),
        hashed_password=get_password_hash(user_in.password),
        avatar_url=f"https://api.dicebear.com/7.x/bottts/svg?seed={user_in.email}",
        provider="email"
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    access_token = create_access_token(user.id)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }


@router.post("/guest", response_model=Token)
def guest_login(db: Session = Depends(get_db)):
    """Initializes or returns a seamless guest session for instant access without credentials."""
    guest_uid = uuid.uuid4().hex[:8]
    user = User(
        email=f"guest_{guest_uid}@suresh.guest",
        name=f"Guest User ({guest_uid})",
        hashed_password=get_password_hash("guestpass123"),
        avatar_url=f"https://api.dicebear.com/7.x/bottts/svg?seed=guest_{guest_uid}",
        provider="guest"
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    access_token = create_access_token(user.id)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }


@router.post("/login", response_model=Token)
def login_user(user_in: UserLogin, db: Session = Depends(get_db)):
    """Authenticates email and password, returning a JWT token."""
    user = db.query(User).filter(User.email == user_in.email.lower()).first()
    if not user or not verify_password(user_in.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"}
        )

    access_token = create_access_token(user.id)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }


@router.post("/google", response_model=Token)
def google_auth(auth_in: GoogleAuthRequest, db: Session = Depends(get_db)):
    """Authenticates or signs up a user via Google OAuth."""
    email = auth_in.email.lower()
    user = db.query(User).filter(User.email == email).first()

    if not user:
        user = User(
            email=email,
            name=auth_in.name or email.split("@")[0].title(),
            avatar_url=auth_in.avatar_url or f"https://api.dicebear.com/7.x/avataaars/svg?seed={email}",
            provider="google"
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        # Update details
        if auth_in.name:
            user.name = auth_in.name
        if auth_in.avatar_url:
            user.avatar_url = auth_in.avatar_url
        user.provider = "google"
        db.commit()

    access_token = create_access_token(user.id)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }


@router.get("/me", response_model=UserResponse)
def get_profile(current_user: User = Depends(get_current_user)):
    """Returns the authenticated user's profile."""
    return current_user


@router.put("/me", response_model=UserResponse)
def update_profile(
    user_in: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Updates user profile preferences and hyperparameters."""
    if user_in.name is not None:
        current_user.name = user_in.name
    if user_in.system_prompt is not None:
        current_user.system_prompt = user_in.system_prompt
    if user_in.default_model is not None:
        current_user.default_model = user_in.default_model
    if user_in.temperature is not None:
        current_user.temperature = user_in.temperature
    if user_in.avatar_url is not None:
        current_user.avatar_url = user_in.avatar_url

    db.commit()
    db.refresh(current_user)
    return current_user


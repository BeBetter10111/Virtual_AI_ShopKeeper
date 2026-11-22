import asyncio
import base64
import hashlib
import re
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import bcrypt
import jwt
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models.domain import ChatSession, User
from app.schemas import contracts as c

JWT_ALGORITHM = "HS256"
BCRYPT_ROUNDS = 12
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(Exception):
    """Missing, invalid or expired credentials (HTTP 401)."""

    code: c.ApiErrorCode = "UNAUTHORIZED"
    status_code: int = 401

    def __init__(self, message: str = "Authentication required.") -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] | None = None


class RegistrationError(Exception):
    """Invalid registration data (HTTP 422)."""

    code: c.ApiErrorCode = "VALIDATION_ERROR"
    status_code: int = 422

    def __init__(self, message: str, field: str) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] | None = {"field": field}


@dataclass(frozen=True)
class TokenClaims:
    user_id: uuid.UUID
    session_id: str


def normalize_email(email: str) -> str:
    return email.strip().lower()


# Passwords 


def _prehash(password: str) -> bytes:
    # SHA-256 first: bcrypt silently ignores bytes beyond 72 and rejects NULs.
    return base64.b64encode(hashlib.sha256(password.encode("utf-8")).digest())


def _hash_sync(password: str) -> str:
    salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    return bcrypt.hashpw(_prehash(password), salt).decode("ascii")


def _verify_sync(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_prehash(password), hashed.encode("ascii"))
    except ValueError:
        return False


async def hash_password(password: str) -> str:
    return await asyncio.to_thread(_hash_sync, password)


async def verify_password(password: str, hashed: str) -> bool:
    return await asyncio.to_thread(_verify_sync, password, hashed)


@lru_cache
def _dummy_hash_for(rounds: int) -> str:
    return bcrypt.hashpw(b"dummy", bcrypt.gensalt(rounds=rounds)).decode("ascii")


# Tokens 


def create_access_token(
    user_id: uuid.UUID,
    session_id: str,
    settings: Settings,
    ttl_seconds: int | None = None,
) -> str:
    now = int(time.time())
    ttl = settings.access_token_ttl_seconds if ttl_seconds is None else ttl_seconds
    claims = {"sub": str(user_id), "sid": session_id, "iat": now, "exp": now + ttl}
    return jwt.encode(claims, settings.jwt_secret_key, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str, settings: Settings) -> TokenClaims:
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["exp", "sub", "sid"]},
        )
        return TokenClaims(
            user_id=uuid.UUID(str(claims["sub"])), session_id=str(claims["sid"])
        )
    except (jwt.PyJWTError, ValueError) as exc:
        raise AuthError("Invalid or expired access token.") from exc


# Sessions 


async def _open_session(
    db: AsyncSession, user: User, settings: Settings
) -> c.SessionResponse:
    chat_session = ChatSession(user_id=user.id)
    db.add(chat_session)
    await db.flush()
    session_id = str(chat_session.id)
    await db.commit()
    return c.SessionResponse(
        access_token=create_access_token(user.id, session_id, settings),
        session_id=session_id,
        user_id=str(user.id),
        expires_in=settings.access_token_ttl_seconds,
    )


async def start_guest_session(
    db: AsyncSession, accessibility_mode: bool, settings: Settings
) -> c.SessionResponse:
    user = User(
        is_guest=True,
        accessibility_prefs={"accessibility_mode": accessibility_mode},
    )
    db.add(user)
    await db.flush()
    return await _open_session(db, user, settings)


async def register_user(
    db: AsyncSession, request: c.RegisterRequest, settings: Settings
) -> c.SessionResponse:
    email = normalize_email(request.email)
    if not EMAIL_PATTERN.match(email):
        raise RegistrationError("Email address is not valid.", "email")
    taken = await db.scalar(select(User.id).where(User.email == email))
    if taken is not None:
        raise RegistrationError("Email is already registered.", "email")

    user = User(
        email=email,
        password_hash=await hash_password(request.password),
        is_guest=False,
        accessibility_prefs={"accessibility_mode": request.accessibility_mode},
    )
    db.add(user)
    try:
        await db.flush()
    except IntegrityError as exc:
        await db.rollback()
        raise RegistrationError("Email is already registered.", "email") from exc
    return await _open_session(db, user, settings)


async def login(
    db: AsyncSession, email: str, password: str, settings: Settings
) -> c.SessionResponse:
    user = await db.scalar(select(User).where(User.email == normalize_email(email)))
    if user is None or user.password_hash is None:
        # Same work as a real check, so response time does not reveal the email.
        await verify_password(password, _dummy_hash_for(BCRYPT_ROUNDS))
        raise AuthError("Invalid email or password.")
    if not await verify_password(password, user.password_hash):
        raise AuthError("Invalid email or password.")
    return await _open_session(db, user, settings)


async def load_user(db: AsyncSession, claims: TokenClaims) -> User:
    user = await db.get(User, claims.user_id)
    if user is None:
        raise AuthError("Invalid or expired access token.")
    return user

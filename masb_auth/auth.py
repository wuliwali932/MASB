import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

from . import users

DEFAULT_AUTH_SECRET = (
    "medical agent security bench smart on fhir login service"

)
SECRET_KEY = os.getenv("MASB_AUTH_SECRET", DEFAULT_AUTH_SECRET)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_SECONDS = 3600  # 1 hour

security = HTTPBearer()


def create_access_token(*, data: dict, expires_delta: Optional[timedelta] = None) -> str:
    payload = data.copy()
    now = datetime.utcnow()
    expire = now + \
        (expires_delta or timedelta(seconds=ACCESS_TOKEN_EXPIRE_SECONDS))
    payload.update({"exp": expire, "iat": now})
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Token has expired", headers={"WWW-Authenticate": "Bearer"})
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Token is invalid", headers={"WWW-Authenticate": "Bearer"})


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_access_token(token)
    username: Optional[str] = payload.get("sub")
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Token missing subject", headers={"WWW-Authenticate": "Bearer"})
    user = await run_in_threadpool(users.get_user, username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.get("disabled"):
        raise HTTPException(status_code=400, detail="User is disabled")
    return user


def check_role(allowed_roles: List[str]):
    """Create a dependency that checks if the current user has one of the allowed roles."""
    async def role_checker(current_user: Dict = Depends(get_current_user)):
        if current_user.get("role") not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: one of {allowed_roles}"
            )
        return current_user
    return role_checker


# Role-specific dependencies
get_patient = check_role(["patient"])
get_physician = check_role(["physician"])
get_administrator = check_role(["administrator"])
get_medical_staff = check_role(["physician", "administrator"])

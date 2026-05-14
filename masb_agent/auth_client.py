"""SMART on FHIR auth client used by the REACT agent."""

import os
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import jwt

logger = logging.getLogger(__name__)

DEFAULT_AUTH_SECRET = (
    "medical agent security bench smart on fhir login service"
    "medical agent security bench smart on fhir login service"
    "medical agent security bench smart on fhir login service"
)


class REACTAuthClient:
    """Authentication client for REACT agent token and role state."""

    def __init__(
        self,
        auth_service_url: str = "http://localhost:8000",
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        token_expire_seconds: int = 3600,
    ) -> None:
        self.auth_service_url = auth_service_url.rstrip('/')
        self.secret_key = secret_key or os.getenv("MASB_AUTH_SECRET", DEFAULT_AUTH_SECRET)
        self.algorithm = algorithm
        self.token_expire_seconds = token_expire_seconds
        self.current_token: Optional[str] = None
        self.current_user: Optional[Dict] = None
        self.token_created_at: Optional[datetime] = None

    def login(self, username: str, password: str) -> Dict[str, Any]:
        """
        Authenticate user and obtain JWT token.

        Args:
            username: User's username
            password: User's password

        Returns:
            Dictionary containing access_token, token_type, expires_in, and user role

        Raises:
            Exception: If authentication fails
        """
        try:
            import requests
            url = f"{self.auth_service_url}/login"
            payload = {"username": username, "password": password}

            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()

            result = response.json()
            self.current_token = result["access_token"]
            self.token_created_at = datetime.utcnow()

            self.current_user = self._decode_token(self.current_token)
            self.current_user.setdefault("username", username)
            if result.get("role"):
                self.current_user["role"] = result["role"]

            logger.info("User '%s' authenticated with role: %s", username, result.get("role"))
            return result

        except requests.exceptions.RequestException as e:
            logger.error("Authentication failed: %s", e)
            raise Exception(f"Authentication service error: {str(e)}")
        except Exception as e:
            logger.error("Login error: %s", e)
            raise

    def _decode_token(self, token: str) -> Dict:
        """Decode JWT token without verification (for local use)."""
        try:
            return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            raise Exception("Token has expired")
        except jwt.InvalidTokenError:
            logger.warning("Token is invalid")
            raise Exception("Token is invalid")

    def get_token(self) -> Optional[str]:
        """Get current valid token."""
        if not self.current_token:
            return None

        if self._is_token_expired():
            logger.warning("Token has expired")
            self.current_token = None
            return None

        return self.current_token

    def _is_token_expired(self) -> bool:
        """Check if current token is expired."""
        if not self.token_created_at or not self.current_token:
            return True

        expiry_time = self.token_created_at + timedelta(seconds=self.token_expire_seconds)
        return datetime.utcnow() >= expiry_time

    def get_auth_header(self) -> Dict[str, str]:
        """Get Authorization header for FHIR API requests."""
        token = self.get_token()
        if not token:
            raise Exception("No valid authentication token. Please login first.")

        return {"Authorization": f"Bearer {token}"}

    def get_current_user_info(self) -> Optional[Dict]:
        """Get current authenticated user information."""
        if self._is_token_expired():
            return None
        return self.current_user

    def verify_role(self, required_role: str) -> bool:
        """Verify if current user has required role."""
        if not self.current_user:
            return False

        user_role = self.current_user.get("role")
        return user_role == required_role or user_role == "administrator"

    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated."""
        return self.get_token() is not None

    def logout(self) -> None:
        """Clear authentication tokens and user info."""
        self.current_token = None
        self.current_user = None
        self.token_created_at = None
        logger.info("User logged out")

    def get_role(self) -> Optional[str]:
        """Get current user's role."""
        return self.current_user.get("role") if self.current_user else None


# Global auth client instance
_auth_client: Optional[REACTAuthClient] = None


def get_react_auth_client() -> REACTAuthClient:
    """Get or create the global REACT auth client instance."""
    global _auth_client
    if _auth_client is None:
        _auth_client = REACTAuthClient()
    return _auth_client


def set_react_auth_client(client: REACTAuthClient) -> None:
    """Set the global REACT auth client instance."""
    global _auth_client
    _auth_client = client

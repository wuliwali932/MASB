"""SMART on FHIR auth manager for the orchestration agent."""

import os
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import jwt
import requests

logger = logging.getLogger(__name__)

DEFAULT_AUTH_SECRET = (
    "medical agent security bench smart on fhir login service"
    "medical agent security bench smart on fhir login service"
    "medical agent security bench smart on fhir login service"
)


class SMARTAuthManager:
    """Manage auth service login state and bearer-token headers."""

    def __init__(
        self,
        auth_service_url: str = "http://localhost:8000",
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        token_expire_seconds: int = 3600,
    ) -> None:
        self.auth_service_url = auth_service_url.rstrip("/")
        self.secret_key = secret_key or os.getenv("MASB_AUTH_SECRET", DEFAULT_AUTH_SECRET)
        self.algorithm = algorithm
        self.token_expire_seconds = token_expire_seconds
        self.current_token: Optional[str] = None
        self.current_user: Optional[Dict] = None
        self.token_created_at: Optional[datetime] = None

    def login(self, username: str, password: str) -> Dict[str, Any]:
        """
        Authenticate user and obtain JWT token from masb_auth service.

        Args:
            username: User's username
            password: User's password

        Returns:
            Dictionary containing access_token, token_type, expires_in, and user role

        Raises:
            Exception: If authentication fails
        """
        try:
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
        """
        Decode JWT token without verification (for local use).

        Args:
            token: JWT token

        Returns:
            Decoded token payload
        """
        try:
            return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            raise Exception("Token has expired")
        except jwt.InvalidTokenError:
            logger.warning("Token is invalid")
            raise Exception("Token is invalid")

    def get_token(self) -> Optional[str]:
        """
        Get current valid token.

        Returns:
            JWT token if valid and not expired, None otherwise
        """
        if not self.current_token:
            return None

        # Check if token is still valid
        if self._is_token_expired():
            logger.warning("Token has expired")
            self.current_token = None
            return None

        return self.current_token

    def _is_token_expired(self) -> bool:
        """
        Check if current token is expired.

        Returns:
            True if token is expired, False otherwise
        """
        if not self.token_created_at or not self.current_token:
            return True

        expiry_time = self.token_created_at + timedelta(seconds=self.token_expire_seconds)
        return datetime.utcnow() >= expiry_time

    def get_auth_header(self) -> Dict[str, str]:
        """
        Get Authorization header for FHIR API requests.

        Returns:
            Dictionary with Authorization header using Bearer token
        """
        token = self.get_token()
        if not token:
            raise Exception("No valid authentication token. Please login first.")

        return {"Authorization": f"Bearer {token}"}

    def get_current_user_info(self) -> Optional[Dict]:
        """
        Get current authenticated user information.

        Returns:
            User info dictionary with username, role, etc.
        """
        if self._is_token_expired():
            return None
        return self.current_user

    def verify_role(self, required_role: str) -> bool:
        """
        Verify if current user has required role.

        Args:
            required_role: Required role name

        Returns:
            True if user has required role, False otherwise
        """
        if not self.current_user:
            return False

        user_role = self.current_user.get("role")
        return user_role == required_role or user_role == "administrator"

    def verify_role_any(self, allowed_roles: list) -> bool:
        """
        Verify if current user has any of the allowed roles.

        Args:
            allowed_roles: List of allowed role names

        Returns:
            True if user has one of the allowed roles, False otherwise
        """
        if not self.current_user:
            return False

        user_role = self.current_user.get("role")
        return user_role in allowed_roles or user_role == "administrator"

    def logout(self) -> None:
        """Clear authentication tokens and user info."""
        self.current_token = None
        self.current_user = None
        self.token_created_at = None
        logger.info("User logged out")

    def get_bearer_token(self) -> str:
        """
        Get JWT token for Bearer authorization.

        Returns:
            JWT token string
        """
        token = self.get_token()
        if not token:
            raise Exception("No valid authentication token. Please login first.")
        return token


class SMART2FAManager(SMARTAuthManager):
    """
    SMART on FHIR Authentication Manager with 2FA support.
    Extends SMARTAuthManager with two-factor authentication capabilities.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize SMART 2FA Auth Manager."""
        super().__init__(*args, **kwargs)
        self.mfa_required: Dict[str, bool] = {}
        self.mfa_verified: Dict[str, bool] = {}

    def login_with_2fa(
        self,
        username: str,
        password: str,
        mfa_code: Optional[str] = None
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Authenticate user with optional 2FA.

        Args:
            username: User's username
            password: User's password
            mfa_code: MFA code (if 2FA is required)

        Returns:
            Tuple of (auth_result, requires_mfa)
        """
        try:
            url = f"{self.auth_service_url}/login"
            payload = {"username": username, "password": password}

            if mfa_code:
                payload["mfa_code"] = mfa_code

            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()

            result = response.json()

            # Check if 2FA is required
            if result.get("requires_mfa"):
                self.mfa_required[username] = True
                logger.info(f"2FA required for user '{username}'")
                return result, True

            # If 2FA passed, set token
            self.current_token = result["access_token"]
            self.token_created_at = datetime.utcnow()
            self.current_user = self._decode_token(self.current_token)
            self.current_user.setdefault("username", username)
            if result.get("role"):
                self.current_user["role"] = result["role"]
            self.mfa_verified[username] = True

            logger.info("User '%s' authenticated successfully with 2FA", username)
            return result, False

        except requests.exceptions.RequestException as e:
            logger.error("Authentication failed: %s", e)
            raise Exception(f"Authentication service error: {str(e)}")


# Global auth manager instance
_auth_manager: Optional[SMARTAuthManager] = None


def get_auth_manager(
    auth_service_url: Optional[str] = None,
    secret_key: Optional[str] = None,
    enable_2fa: bool = False
) -> SMARTAuthManager:
    """
    Get or create global auth manager instance.

    Args:
        auth_service_url: URL of masb_auth service
        secret_key: Secret key for JWT
        enable_2fa: Enable 2FA support

    Returns:
        SMARTAuthManager instance
    """
    global _auth_manager

    if _auth_manager is None:
        auth_service_url = auth_service_url or os.getenv(
            "MASB_AUTH_URL",
            "http://localhost:8000"
        )
        secret_key = secret_key or os.getenv("MASB_AUTH_SECRET")

        if enable_2fa:
            _auth_manager = SMART2FAManager(
                auth_service_url=auth_service_url,
                secret_key=secret_key
            )
        else:
            _auth_manager = SMARTAuthManager(
                auth_service_url=auth_service_url,
                secret_key=secret_key
            )

    return _auth_manager


# Usage Examples

if __name__ == "__main__":
    # Example 1: Basic authentication
    print("Example 1: Basic Authentication")
    auth = SMARTAuthManager(auth_service_url="http://localhost:8000")

    try:
        result = auth.login("doctor1", "d1secret")
        print(f"Login successful: {result}")

        # Get current user info
        user_info = auth.get_current_user_info()
        print(f"Current user: {user_info}")

        # Verify role
        if auth.verify_role("physician"):
            print("User has physician role")

        # Get auth header for FHIR requests
        headers = auth.get_auth_header()
        print(f"Auth header: {headers}")

        # Logout
        auth.logout()
        print("Logged out successfully")

    except Exception as e:
        print(f"Error: {e}")

    # Example 2: 2FA authentication
    print("\nExample 2: 2FA Authentication")
    auth_2fa = SMART2FAManager(auth_service_url="http://localhost:8000")

    try:
        # First attempt without MFA code
        result, requires_mfa = auth_2fa.login_with_2fa(
            "doctor1", "d1secret")

        if requires_mfa:
            print("2FA required. Sending MFA code...")
            mfa_code = input("Enter MFA code: ")

            # Retry with MFA code
            result, requires_mfa = auth_2fa.login_with_2fa(
                "doctor1",
                "d1secret",
                mfa_code=mfa_code
            )

            if not requires_mfa:
                print(f"2FA login successful: {result}")

    except Exception as e:
        print(f"Error: {e}")

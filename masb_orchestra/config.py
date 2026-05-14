"""
Configuration template for Medical Agent Orchestration System
Copy this file and customize for your deployment environment.
"""

import os
from typing import Optional


class Config:
    """Base configuration class."""

    # ====================
    # LLM Configuration
    # ====================

    # Choose one: "openai", "google", "anthropic"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")

    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4")
    OPENAI_TEMPERATURE: float = 0.0  # Use 0 for deterministic medical decisions
    OPENAI_MAX_TOKENS: int = 4096

    # Google Gemini Configuration
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_MODEL: str = "gemini-pro"

    # Anthropic Configuration
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = "claude-3-opus-20240229"

    # ====================
    # FHIR Server Configuration
    # ====================

    # FHIR Server URL
    FHIR_SERVER_URL: str = os.getenv(
        "FHIR_SERVER_URL",
        "http://localhost:8080/fhir"
    )

    # FHIR Authentication (if needed)
    FHIR_AUTH_TYPE: str = os.getenv(
        "FHIR_AUTH_TYPE", "none")  # "none", "basic", "oauth2"
    FHIR_USERNAME: Optional[str] = os.getenv("FHIR_USERNAME")
    FHIR_PASSWORD: Optional[str] = os.getenv("FHIR_PASSWORD")
    FHIR_CLIENT_ID: Optional[str] = os.getenv("FHIR_CLIENT_ID")
    FHIR_CLIENT_SECRET: Optional[str] = os.getenv("FHIR_CLIENT_SECRET")

    # FHIR Server Options
    FHIR_REQUEST_TIMEOUT: int = 10  # seconds
    FHIR_VERIFY_SSL: bool = True

    # ====================
    # Application Configuration
    # ====================

    # Environment
    # development, staging, production
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    # Logging
    # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE", "medical_orchestrator.log")

    # Execution
    MAX_EXECUTION_TIME: int = 300  # seconds
    ENABLE_TASK_DECOMPOSITION: bool = True
    ENABLE_EXECUTION_LOGGING: bool = True

    # ====================
    # Medical Data Configuration
    # ====================

    # Lab Codes (Customizable)
    LAB_CODES: dict = {
        "MG": "Magnesium",
        "K": "Potassium",
        "GLU": "Glucose (Blood)",
        "A1C": "Hemoglobin A1C",
        "BP": "Blood Pressure",
        "QTINTERVAL": "QT Interval",
        "Cr": "Creatinine"
    }

    # Reference Ranges (Customizable)
    REFERENCE_RANGES: dict = {
        "MG": {
            "normal_low": 1.7,
            "normal_high": 2.2,
            "units": "mg/dL"
        },
        "K": {
            "normal_low": 3.5,
            "normal_high": 5.0,
            "units": "mEq/L"
        },
        "GLU": {
            "normal_low": 70,
            "normal_high": 100,
            "units": "mg/dL"
        },
        "A1C": {
            "normal_high": 5.7,
            "units": "%"
        }
    }

    # Medication Information
    MEDICATIONS: dict = {
        "magnesium": {
            "ndc": "0338-1715-40",
            "type": "IV"
        },
        "potassium": {
            "ndc": "40032-917-01",
            "type": "oral"
        }
    }

    # ====================
    # Security Configuration
    # ====================

    # Enable audit logging for all operations
    ENABLE_AUDIT_LOG: bool = True
    AUDIT_LOG_FILE: str = "audit.log"

    # Enable role-based access control
    ENABLE_RBAC: bool = False

    # Allowed operations for different roles
    ROLE_PERMISSIONS: dict = {
        "physician": ["read", "write", "calculate"],
        "nurse": ["read", "write"],
        "pharmacist": ["read", "calculate"],
        "admin": ["read", "write", "calculate", "delete"]
    }

    # ====================
    # External Tool Configuration
    # ====================

    # Enable/disable specific tools
    ENABLE_LAB_REFERENCE: bool = True
    ENABLE_DOSING_CALCULATOR: bool = True
    ENABLE_DRUG_INTERACTION_CHECK: bool = True
    ENABLE_ALLERGY_CHECK: bool = True
    ENABLE_NOTIFICATIONS: bool = True

    # Notification Configuration
    NOTIFICATION_SERVICE: str = "email"  # "email", "sms", "slack"
    NOTIFICATION_EMAIL_FROM: str = "medical-system@hospital.org"
    NOTIFICATION_SLACK_WEBHOOK: str = os.getenv("SLACK_WEBHOOK", "")

    # ====================
    # Performance Configuration
    # ====================

    # Caching
    ENABLE_CACHING: bool = True
    CACHE_TTL: int = 3600  # seconds

    # Parallel Execution
    ENABLE_PARALLEL_SUBTASKS: bool = True
    MAX_PARALLEL_TASKS: int = 5

    # ====================
    # Development Configuration
    # ====================

    # Mock FHIR Server
    USE_MOCK_FHIR: bool = False  # Use mock for testing without FHIR server

    # Mock Data
    MOCK_PATIENTS: dict = {
        "S6534835": {
            "name": "Peter Stafford",
            "dob": "1932-12-29",
            "gender": "male"
        },
        "S6315806": {
            "name": "Test Patient",
            "dob": "1970-01-01",
            "gender": "male"
        }
    }

    # ====================
    # Method Overrides
    # ====================

    @classmethod
    def get_llm_config(cls) -> dict:
        """Get LLM configuration based on provider."""
        if cls.LLM_PROVIDER == "openai":
            return {
                "provider": "openai",
                "model": cls.OPENAI_MODEL,
                "api_key": cls.OPENAI_API_KEY,
                "temperature": cls.OPENAI_TEMPERATURE,
                "max_tokens": cls.OPENAI_MAX_TOKENS
            }
        elif cls.LLM_PROVIDER == "google":
            return {
                "provider": "google",
                "model": cls.GOOGLE_MODEL,
                "api_key": cls.GOOGLE_API_KEY
            }
        elif cls.LLM_PROVIDER == "anthropic":
            return {
                "provider": "anthropic",
                "model": cls.ANTHROPIC_MODEL,
                "api_key": cls.ANTHROPIC_API_KEY
            }
        else:
            raise ValueError(f"Unknown LLM provider: {cls.LLM_PROVIDER}")

    @classmethod
    def get_fhir_config(cls) -> dict:
        """Get FHIR server configuration."""
        return {
            "server_url": cls.FHIR_SERVER_URL,
            "auth_type": cls.FHIR_AUTH_TYPE,
            "username": cls.FHIR_USERNAME,
            "password": cls.FHIR_PASSWORD,
            "timeout": cls.FHIR_REQUEST_TIMEOUT,
            "verify_ssl": cls.FHIR_VERIFY_SSL
        }

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration."""
        errors = []

        # Validate LLM configuration
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required for OpenAI provider")

        elif cls.LLM_PROVIDER == "google" and not cls.GOOGLE_API_KEY:
            errors.append("GOOGLE_API_KEY is required for Google provider")

        elif cls.LLM_PROVIDER == "anthropic" and not cls.ANTHROPIC_API_KEY:
            errors.append(
                "ANTHROPIC_API_KEY is required for Anthropic provider")

        # Validate FHIR configuration
        if not cls.FHIR_SERVER_URL and not cls.USE_MOCK_FHIR:
            errors.append(
                "FHIR_SERVER_URL is required or USE_MOCK_FHIR must be True")

        if errors:
            print("Configuration Errors:")
            for error in errors:
                print(f"  - {error}")
            return False

        return True


class DevelopmentConfig(Config):
    """Development environment configuration."""
    ENVIRONMENT = "development"
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    USE_MOCK_FHIR = True
    ENABLE_RBAC = False
    OPENAI_TEMPERATURE = 0.0


class StagingConfig(Config):
    """Staging environment configuration."""
    ENVIRONMENT = "staging"
    DEBUG = False
    LOG_LEVEL = "INFO"
    USE_MOCK_FHIR = False
    ENABLE_RBAC = True
    FHIR_VERIFY_SSL = True


class ProductionConfig(Config):
    """Production environment configuration."""
    ENVIRONMENT = "production"
    DEBUG = False
    LOG_LEVEL = "WARNING"
    USE_MOCK_FHIR = False
    ENABLE_RBAC = True
    ENABLE_AUDIT_LOG = True
    FHIR_VERIFY_SSL = True
    MAX_EXECUTION_TIME = 60  # Shorter timeout for production
    ENABLE_PARALLEL_SUBTASKS = True
    MAX_PARALLEL_TASKS = 10


def get_config() -> Config:
    """Get configuration based on environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()

    if env == "production":
        return ProductionConfig
    elif env == "staging":
        return StagingConfig
    else:
        return DevelopmentConfig


# Usage Example
if __name__ == "__main__":
    # Get configuration for current environment
    config = get_config()

    # Validate configuration
    if not config.validate():
        exit(1)

    # Print configuration
    print(f"Environment: {config.ENVIRONMENT}")
    print(f"LLM Provider: {config.LLM_PROVIDER}")
    print(f"FHIR Server: {config.FHIR_SERVER_URL}")
    print(f"Debug Mode: {config.DEBUG}")

    # Get specific configs
    llm_config = config.get_llm_config()
    fhir_config = config.get_fhir_config()

    print("\nLLM Configuration:")
    for key, value in llm_config.items():
        if key == "api_key":
            print(f"  {key}: {'*' * 20}")
        else:
            print(f"  {key}: {value}")

    print("\nFHIR Configuration:")
    for key, value in fhir_config.items():
        if key in ["password"]:
            print(f"  {key}: {'*' * 20}")
        else:
            print(f"  {key}: {value}")

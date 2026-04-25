"""
server/config.py — Centralized configuration via environment variables.
All hardcoded constants live here. Override via .env or container env vars.
"""
import os


class Config:
    # Episode settings
    MAX_BUDGET: int = int(os.getenv("MAX_BUDGET", "12"))
    SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", "1800"))  # 30 min
    MAX_CONCURRENT_SESSIONS: int = int(os.getenv("MAX_CONCURRENT_SESSIONS", "50"))

    # Anti-exploit
    MAX_SAME_ACTION_STREAK: int = int(os.getenv("MAX_SAME_ACTION_STREAK", "3"))
    STEP_TIMEOUT_SECONDS: int = int(os.getenv("STEP_TIMEOUT_SECONDS", "10"))

    # Curriculum thresholds
    CURRICULUM_MEDIUM_AFTER: int = int(os.getenv("CURRICULUM_MEDIUM_AFTER", "20"))
    CURRICULUM_HARD_AFTER: int = int(os.getenv("CURRICULUM_HARD_AFTER", "50"))

    # Dataset
    DATASET_N_SAMPLES: int = int(os.getenv("DATASET_N_SAMPLES", "600"))
    GOLDEN_ROW_COUNT: int = int(os.getenv("GOLDEN_ROW_COUNT", "10"))

    # Server
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENV_VERSION: str = "0.3.0"


cfg = Config()

import logging

from abcfold.backend_envs import BackendEnv, ensure_backend_env

logger = logging.getLogger("logger")


def ensure_boltz_env(config: dict) -> BackendEnv:
    """Verify (never modify) the pinned boltz environment."""
    return ensure_backend_env("boltz", config)

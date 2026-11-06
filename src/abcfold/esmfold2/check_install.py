import logging

from abcfold.backend_envs import BackendEnv, ensure_backend_env

logger = logging.getLogger("logger")


def ensure_esmfold2_env(config: dict) -> BackendEnv:
    """Verify (never modify) the pinned ESMFold2 environment."""
    return ensure_backend_env("esmfold2", config)

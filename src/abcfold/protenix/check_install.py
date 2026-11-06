import logging

from abcfold.backend_envs import BackendEnv, ensure_backend_env

logger = logging.getLogger("logger")


def ensure_protenix_env(config: dict) -> BackendEnv:
    """Verify (never modify) the pinned protenix environment."""
    return ensure_backend_env("protenix", config)

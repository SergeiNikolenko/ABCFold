import logging

from abcfold.backend_envs import BackendEnv, ensure_backend_env

logger = logging.getLogger("logger")


def ensure_opendde_env(config: dict) -> BackendEnv:
    """Verify (never modify) the pinned OpenDDE environment."""
    return ensure_backend_env("opendde", config)

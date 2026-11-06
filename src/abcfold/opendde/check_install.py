import logging

from abcfold.backend_envs import MicromambaEnv, ensure_backend_env

logger = logging.getLogger("logger")


def ensure_opendde_env(config: dict) -> MicromambaEnv:
    """Verify (never modify) the pinned OpenDDE environment."""
    return ensure_backend_env("opendde", config)

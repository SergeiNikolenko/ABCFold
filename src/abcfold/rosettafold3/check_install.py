import logging
import urllib.request
from pathlib import Path

from abcfold.backend_envs import BackendEnv, ensure_backend_env

logger = logging.getLogger("logger")

RF3_BASE_URL = "http://files.ipd.uw.edu/pub/rf3"
CHECKPOINT_NAME = "rf3_foundry_01_24_latest_remapped.ckpt"
RF3_URL = f"{RF3_BASE_URL}/{CHECKPOINT_NAME}"


def ensure_rosettafold_checkpoint(target_path: Path) -> Path:
    if target_path.exists():
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(
            "Downloading RoseTTAFold3 checkpoint via HTTPS "
            "(this may take a while)..."
        )
        urllib.request.urlretrieve(RF3_URL, target_path)
    except Exception as e:
        raise RuntimeError(
            "Failed to download RoseTTAFold3 checkpoint.\n"
            f"Target: {target_path}\n"
            f"Error: {e}"
        )

    if not target_path.exists():
        raise RuntimeError("Checkpoint download completed but file not found")

    return target_path


def ensure_rosettafold_env(config: dict) -> BackendEnv:
    """Verify (never modify) the pinned RoseTTAFold3 environment."""
    return ensure_backend_env("rosettafold3", config)

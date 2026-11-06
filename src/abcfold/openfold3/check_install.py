import logging
import os
from pathlib import Path
from typing import Optional

from abcfold.backend_envs import BackendEnv, ensure_backend_env

logger = logging.getLogger("logger")

DEFAULT_CHECKPOINT = "openbind-2025-06-30-174k"


def ensure_openfold_env(config: dict) -> BackendEnv:
    """Verify (never modify) the pinned OpenFold3 environment."""
    return ensure_backend_env("openfold3", config)


def openfold_cache_dir(config: dict) -> Path:
    weights = config.get("openfold_weights")
    if weights:
        return Path(str(weights)).expanduser()
    return Path(os.environ.get("OPENFOLD_CACHE", Path.home() / ".openfold3"))


def ensure_openfold_checkpoint(
    env: BackendEnv,
    cache_dir: Path,
    checkpoint_name: Optional[str] = None,
) -> None:
    """Download the pinned OpenFold3 parameters into ``cache_dir`` if missing.

    Uses OpenFold3's own downloader inside the backend env, so the checkpoint
    registry always matches the installed OpenFold3 version.
    """
    name = checkpoint_name or DEFAULT_CHECKPOINT
    code = (
        "from pathlib import Path\n"
        "from openfold3.entry_points.parameters import (\n"
        "    OPENFOLD_MODEL_CHECKPOINT_REGISTRY, download_model_parameters,\n"
        "    get_default_checkpoint_dir)\n"
        f"d = get_default_checkpoint_dir(cache_path=Path({str(cache_dir)!r}))\n"
        f"name = {name!r}\n"
        "f = d / OPENFOLD_MODEL_CHECKPOINT_REGISTRY[name].file_name\n"
        "if not f.exists():\n"
        "    download_model_parameters(d, name, skip_confirmation=True)\n"
        "print(f)\n"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    env.run(["python", "-c", code], quiet=True)

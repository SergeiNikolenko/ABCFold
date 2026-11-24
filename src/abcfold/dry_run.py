"""Set up and verify ABCFold predictor environments without running inference.

Supports ``abcfold <input_json> <output_dir> -b -c ... --dry_run`` and the
standalone ``python -m abcfold.dry_run -b -c ...`` entry point.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from abcfold.backend_envs import (BackendEnv, BackendEnvSpec, ContainerEnv,
                                  MicromambaEnv,
                                  build_env, container_binds, pull_image,
                                  smoke_command)
from abcfold.settings import BACKEND_NAMES, Settings, load_settings

logger = logging.getLogger("logger")

BACKEND_LABELS = {
    "alphafold3": "AlphaFold3",
    "boltz": "Boltz",
    "chai1": "Chai-1",
    "protenix": "Protenix",
    "openfold3": "OpenFold3",
    "rosettafold3": "RoseTTAFold3",
    "intellifold": "IntelliFold",
    "opendde": "OpenDDE",
    "esmfold2": "ESMFold2",
}


def setup_environments(args, settings: Optional[Settings] = None) -> list[str]:
    """Prepare and smoke-check every selected backend without inference."""
    if settings is None:
        settings = load_settings(getattr(args, "config_file", None))
    rt_config = settings.runtime_config()
    selected: list[str] = []

    for name in BACKEND_NAMES:
        if not getattr(args, name, False):
            continue
        selected.append(BACKEND_LABELS.get(name, name))
        if name == "alphafold3":
            logger.info("[dry-run] AlphaFold3 uses an external runner (%s)",
                        rt_config.get("af3_runner") or "docker")
            continue
        pins = settings.backend(name)
        spec = BackendEnvSpec.from_settings(name, pins)
        logger.info("[dry-run] Checking %s environment...",
                    BACKEND_LABELS.get(name, name))
        env: BackendEnv
        if spec.uses_container:
            container = ContainerEnv(
                spec.image or "", spec.container_runner, container_binds(settings)
            )
            if not container.env_exists():
                pull_image(spec)
            container.run(smoke_command(spec), quiet=True)
            env = container
        else:
            mm_env = MicromambaEnv(spec.env)
            if not mm_env.is_current and not mm_env.env_exists():
                build_env(spec, strip=settings.get("strip_unused_wheels", True))
            mm_env.run(smoke_command(spec), quiet=True)
            env = mm_env

        if name == "openfold3":
            from abcfold.openfold3.check_install import (ensure_openfold_checkpoint,
                                                       openfold_cache_dir)

            cache_dir = openfold_cache_dir(rt_config)
            ensure_openfold_checkpoint(
                env, cache_dir, rt_config.get("openfold_checkpoint")
            )
        elif name == "rosettafold3":
            from abcfold.rosettafold3.check_install import (
                CHECKPOINT_NAME, ensure_rosettafold_checkpoint,
            )

            rf_dir = rt_config.get("rosettafold_weights")
            cache_dir = (
                Path(str(rf_dir)).expanduser()
                if rf_dir
                else Path.home() / ".rosettafold3"
            )
            ensure_rosettafold_checkpoint(cache_dir / CHECKPOINT_NAME)

    if not selected:
        logger.warning(
            "[dry-run] No predictors selected -- nothing to set up. Pass one or "
            "more predictor flags (-a/-b/-c/-p/-o/-r/--intellifold/--opendde/"
            "--esmfold2) to choose which environments to check."
        )
        return []

    logger.info(
        "[dry-run] Setup complete for: %s. Environments are ready; "
        "no inference was run.",
        ", ".join(selected),
    )
    return selected


def main(argv: Optional[list[str]] = None) -> int:
    """Standalone entry point: ``python -m abcfold.dry_run -b -c``."""
    from abcfold.argparse_utils import (alphafold_argparse_util,
                                        boltz_argparse_util,
                                        chai_argparse_util,
                                        esmfold2_argparse_util,
                                        intellifold_argparse_util,
                                        opendde_argparse_util,
                                        openfold_argparse_util,
                                        protenix_argparse_util,
                                        rosettafold_argparse_util)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Set up and verify ABCFold predictor environments without running "
            "inference. No GPU required."
        ),
    )
    parser.add_argument(
        "--config-file",
        dest="config_file",
        default=None,
        help="Optional TOML/INI config file (defaults to packaged settings).",
    )
    for util in (
        alphafold_argparse_util,
        boltz_argparse_util,
        chai_argparse_util,
        openfold_argparse_util,
        protenix_argparse_util,
        rosettafold_argparse_util,
        intellifold_argparse_util,
        opendde_argparse_util,
        esmfold2_argparse_util,
    ):
        parser = util(parser)
    args = parser.parse_args(argv)
    settings = load_settings(args.config_file)
    setup_environments(args, settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

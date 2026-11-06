"""GPU selection relative to the GPUs this process was given."""

import os
from typing import Dict, Mapping, Optional


def to_physical(
    gpus: Optional[str], environ: Optional[Mapping] = None
) -> Optional[str]:
    """Translate ``--gpus`` indices into ids for a child ``CUDA_VISIBLE_DEVICES``.

    ``--gpus 0`` means the first GPU *visible to ABCFold*. Setting
    ``CUDA_VISIBLE_DEVICES=0`` in a child process would instead select physical
    GPU 0 and escape a Slurm or manual allocation such as
    ``CUDA_VISIBLE_DEVICES=2``, so indices are mapped through the inherited mask.

    Returns None for "all"/None (inherit the parent mask unchanged) and "" for "cpu".
    """
    if gpus is None:
        return None
    gpus = str(gpus).strip()
    if gpus.lower() == "all":
        return None
    if gpus.lower() == "cpu":
        return ""
    environ = os.environ if environ is None else environ
    parent = environ.get("CUDA_VISIBLE_DEVICES")
    wanted = [g.strip() for g in gpus.split(",") if g.strip()]
    if parent is None:
        return ",".join(wanted)
    visible = [g.strip() for g in parent.split(",") if g.strip()]
    out = []
    for g in wanted:
        if not g.isdigit() or int(g) >= len(visible):
            raise ValueError(
                f"--gpus {gpus}: index {g} is outside the GPUs given to ABCFold "
                f"(CUDA_VISIBLE_DEVICES={parent})"
            )
        out.append(visible[int(g)])
    return ",".join(out)


def docker_gpu_flags(
    gpus: Optional[str], environ: Optional[Mapping] = None
) -> list:
    """``docker run --gpus`` flags; docker sees host GPUs, so physical ids are
    needed."""
    if gpus is not None and str(gpus).strip().lower() == "cpu":
        return []
    environ = os.environ if environ is None else environ
    physical = to_physical(gpus, environ)
    if physical is None:
        physical = environ.get("CUDA_VISIBLE_DEVICES")
    return ["--gpus", f"device={physical}"] if physical else ["--gpus", "all"]


def cuda_env(
    gpus: Optional[str], environ: Optional[Mapping] = None
) -> Dict[str, str]:
    """Child environment pinning a backend to the ``--gpus`` selection.

    Empty for "all"/None, so the child inherits the parent mask unchanged.
    """
    physical = to_physical(gpus, environ)
    return {} if physical is None else {"CUDA_VISIBLE_DEVICES": physical}

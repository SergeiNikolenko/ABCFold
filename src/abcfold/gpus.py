"""GPU selection relative to the GPUs this process was given."""

import logging
import os
import subprocess
import time
from typing import Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger("logger")

_GPU_MEM_CACHE: Optional[Tuple[float, List[Tuple[str, int, int]]]] = None
_GPU_MEM_CACHE_TTL_SEC = 15.0

# Conservative per-worker VRAM budgets (MiB) used for automatic intra-GPU
# worker packing on high-VRAM GPUs (40/80 GiB). JAX backends (AF3/IntelliFold)
# preallocate ~95% of VRAM and always run 1 worker per GPU.
_ENGINE_VRAM_BUDGET_MIB: Dict[str, Tuple[int, int, int]] = {
    # (<= 400 tokens, <= 700 tokens, > 700 tokens)
    "boltz": (14500, 20000, 32000),
    "chai1": (15000, 22000, 34000),
    "esmfold2": (10000, 16000, 26000),
    "protenix": (18000, 26000, 38000),
    "openfold3": (18000, 26000, 38000),
    "opendde": (16000, 24000, 36000),
    "rosettafold3": (18000, 26000, 38000),
}
_SINGLE_WORKER_ONLY_ENGINES = frozenset({"alphafold3", "intellifold"})
_PACKING_HEADROOM_MIB = 4096
_OPENFOLD_HIGH_VRAM_MIB = 30000


def clear_gpu_cache() -> None:
    """Clear the cached nvidia-smi memory query (useful in tests)."""
    global _GPU_MEM_CACHE
    _GPU_MEM_CACHE = None


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


def fast_torch_env(environ: Optional[Mapping] = None) -> Dict[str, str]:
    """Return low-risk PyTorch/CUDA runtime defaults not already set in env.

    expandable_segments:True reduces CUDA allocator fragmentation across
    variable-length batches and concurrent GPU workers.
    """
    env = os.environ if environ is None else environ
    defaults = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "false",
    }
    return {k: v for k, v in defaults.items() if k not in env}


def _query_raw_nvidia_smi() -> List[Tuple[str, int, int]]:
    """Query physical (index, total_mib, free_mib) rows via nvidia-smi."""
    global _GPU_MEM_CACHE
    now = time.monotonic()
    if (
        _GPU_MEM_CACHE is not None
        and (now - _GPU_MEM_CACHE[0]) < _GPU_MEM_CACHE_TTL_SEC
    ):
        return list(_GPU_MEM_CACHE[1])
    rows: List[Tuple[str, int, int]] = []
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if proc.returncode == 0:
            for raw_line in proc.stdout.splitlines():
                parts = [p.strip() for p in raw_line.split(",")]
                if len(parts) < 3:
                    continue
                try:
                    rows.append(
                        (parts[0], int(float(parts[1])), int(float(parts[2])))
                    )
                except ValueError:
                    continue
    except (OSError, subprocess.SubprocessError):
        pass
    _GPU_MEM_CACHE = (now, list(rows))
    return rows


def query_gpu_memory(
    environ: Optional[Mapping] = None,
    *,
    use_cache: bool = True,
) -> List[Tuple[str, int, int]]:
    """Return [(visible_slot, total_mib, free_mib), ...] for visible GPUs.

    Visible slot indices are 0-based relative to CUDA_VISIBLE_DEVICES when
    set, so passing a slot back into to_physical selects the exact physical
    GPU without escaping a Slurm/manual allocation.
    """
    if not use_cache:
        clear_gpu_cache()
    env = os.environ if environ is None else environ
    parent = env.get("CUDA_VISIBLE_DEVICES")
    raw = _query_raw_nvidia_smi()
    by_phys = {idx: (total, free) for idx, total, free in raw}

    if parent is not None:
        visible = [g.strip() for g in parent.split(",") if g.strip()]
        out: List[Tuple[str, int, int]] = []
        for vis_idx, phys_id in enumerate(visible):
            total, free = by_phys.get(phys_id, (0, 0))
            out.append((str(vis_idx), total, free))
        return out

    return [(idx, total, free) for idx, total, free in raw]


def detect_gpu_slots(
    gpus: Optional[str], environ: Optional[Mapping] = None
) -> List[str]:
    """Expand --gpus into individual visible GPU slots.

    Discovers all GPUs visible to ABCFold (respecting CUDA_VISIBLE_DEVICES
    first, then nvidia-smi) so multi-GPU nodes are saturated automatically.
    """
    env = os.environ if environ is None else environ
    if gpus is not None:
        text = str(gpus).strip()
        lower = text.lower()
        if lower == "cpu":
            return ["cpu"]
        if lower != "all":
            wanted = [g.strip() for g in text.split(",") if g.strip()]
            if wanted:
                to_physical(",".join(wanted), env)
                return wanted

    parent = env.get("CUDA_VISIBLE_DEVICES")
    if parent is not None:
        visible = [g.strip() for g in parent.split(",") if g.strip()]
        if not visible:
            return ["cpu"]
        return [str(i) for i in range(len(visible))]

    raw = _query_raw_nvidia_smi()
    if raw:
        return [idx for idx, _, _ in raw]
    return ["all"]


def gpu_vram_mib(
    gpus: Optional[str] = None, environ: Optional[Mapping] = None
) -> Optional[int]:
    """Return minimum total VRAM (MiB) across the selected GPU(s), or None."""
    env = os.environ if environ is None else environ
    if gpus is not None and str(gpus).strip().lower() == "cpu":
        return None
    slots = detect_gpu_slots(gpus, env)
    if not slots or slots == ["cpu"]:
        return None
    mem_map = {slot: total for slot, total, _ in query_gpu_memory(env) if total > 0}
    if not mem_map:
        return None
    if slots == ["all"]:
        return min(mem_map.values())
    matched = [mem_map[s] for s in slots if s in mem_map]
    return min(matched) if matched else None


def resolve_openfold_low_mem(
    gpus: Optional[str] = None,
    explicit: Optional[bool] = None,
    environ: Optional[Mapping] = None,
) -> bool:
    """Disable OpenFold3 low_mem preset automatically on >=30 GiB GPUs."""
    if explicit is not None:
        return bool(explicit)
    vram = gpu_vram_mib(gpus, environ)
    if vram is None:
        return True
    return vram < _OPENFOLD_HIGH_VRAM_MIB


def _engine_budget_mib(engine: str, est_tokens: int) -> int:
    small, med, large = _ENGINE_VRAM_BUDGET_MIB.get(engine, (18000, 26000, 38000))
    if est_tokens <= 400:
        return small
    if est_tokens <= 700:
        return med
    return large


def plan_gpu_workers(
    gpus: Optional[str],
    num_tasks: int,
    *,
    engine: str = "boltz",
    est_tokens: int = 300,
    workers_per_gpu: Optional[int] = None,
    min_tasks_for_packing: int = 4,
    environ: Optional[Mapping] = None,
) -> List[str]:
    """Plan GPU worker slots for num_tasks parallelizable jobs.

    Expands gpus across all visible physical GPUs (sorted by free VRAM
    descending) and packs up to 2-3 concurrent workers per GPU on large
    40/80 GiB GPUs when free VRAM covers the engine budget plus headroom
    and num_tasks is large enough to amortize loading extra weights.
    """
    env = os.environ if environ is None else environ
    slots = detect_gpu_slots(gpus, env)
    if not slots or slots == ["cpu"] or slots == ["all"] or num_tasks <= 1:
        return slots[:1] if slots else ["all"]

    mem_by_slot = {
        slot: (total, free) for slot, total, free in query_gpu_memory(env)
    }
    ordered = sorted(
        slots,
        key=lambda s: (-(mem_by_slot.get(s, (0, 0))[1]), int(s) if s.isdigit() else 0),
    )

    eff_wpg = workers_per_gpu
    if eff_wpg is None:
        env_val = env.get("ABCFOLD_WORKERS_PER_GPU")
        if env_val is not None and str(env_val).strip().isdigit():
            eff_wpg = max(1, int(str(env_val).strip()))

    if (
        engine in _SINGLE_WORKER_ONLY_ENGINES
        or eff_wpg == 1
        or num_tasks < max(min_tasks_for_packing, 2 * len(ordered))
    ):
        return ordered[: min(len(ordered), num_tasks)]

    budget = _engine_budget_mib(engine, est_tokens)
    per_gpu_cap: Dict[str, int] = {}
    for s in ordered:
        if eff_wpg is not None and eff_wpg > 1:
            per_gpu_cap[s] = min(eff_wpg, 4)
        else:
            _, free_mib = mem_by_slot.get(s, (0, 0))
            if free_mib > budget + _PACKING_HEADROOM_MIB:
                fit = (free_mib - _PACKING_HEADROOM_MIB) // budget
                per_gpu_cap[s] = max(1, min(3, int(fit)))
            else:
                per_gpu_cap[s] = 1

    planned: List[str] = []
    max_rounds = max(per_gpu_cap.values(), default=1)
    for r in range(1, max_rounds + 1):
        for s in ordered:
            if per_gpu_cap[s] >= r:
                planned.append(s)
                if len(planned) >= num_tasks:
                    return planned
    return planned

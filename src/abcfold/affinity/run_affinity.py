"""Batch Boltz-2 affinity scoring of predicted models for ABCFold.

``boltz_affinity`` needs the boltz micromamba env + GPU, so it can't run
in-process in the base ABCFold env. This helper invokes the ``boltz_affinity``
multi-model scorer as a subprocess -- it self-dispatches into the boltz env,
scores every model against one shared MSA in a single inference pass, and
writes a combined CSV. We read that back into a ``{model_path: scores}`` lookup
the HTML/report code can join against per model.
"""

import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from abcfold.gpus import detect_gpu_slots, fast_torch_env

logger = logging.getLogger("logger")


def extract_affinity_msa(
    input_params: dict, output_dir: Union[str, Path]
) -> Optional[str]:
    """Return an .a3m path to reuse for affinity scoring, or None to fetch.

    Args:
        input_params: The input JSON dict for the run.
        output_dir: The ABCFold output directory where the affinity subdir is.

    Returns:
        Path to an .a3m MSA file to reuse for affinity scoring, or None
    """
    proteins = [
        s["protein"]
        for s in input_params.get("sequences", [])
        if isinstance(s, dict) and "protein" in s
    ]
    if len(proteins) != 1:
        return None

    prot = proteins[0]
    if prot.get("unpairedMsaPath"):
        return str(prot["unpairedMsaPath"])
    if prot.get("unpairedMsa"):
        msa_file = Path(output_dir) / "affinity" / "reused_msa.a3m"
        msa_file.parent.mkdir(parents=True, exist_ok=True)
        msa_file.write_text(prot["unpairedMsa"])
        logger.info("Reusing ABCFold's MSA for affinity scoring: %s", msa_file)
        return str(msa_file)
    return None


def run_boltz_affinity(
    model_paths: List[Union[str, Path]],
    output_dir: Union[str, Path],
    smiles: Optional[str] = None,
    ccd: Optional[str] = None,
    ligand_chain: Optional[str] = None,
    msa: Optional[Union[str, Path]] = None,
    config: Optional[dict] = None,
    gpus: Optional[str] = None,
) -> Dict[str, dict]:
    """Score every model with the Boltz-2 affinity head in one batched pass.

    Args:
        model_paths: Predicted complex structures (CIF/PDB), all the same
            protein + ligand.
        output_dir: Where to put the ``affinity/`` working dir and results CSV.
        smiles: Ligand SMILES (one of ``smiles``/``ccd`` is required to score).
        ccd: Ligand CCD code (alternative to ``smiles``).
        ligand_chain: Optional override for which chain holds the ligand.
        msa: Optional precomputed ``.a3m`` reused across all models.

    Returns:
        ``{resolved_model_path: {"affinity_pred_value": float,
        "affinity_probability_binary": float}}``. Empty on failure or when no
        ligand chemistry is given, so callers can proceed without scores.
    """
    paths = [Path(p).resolve() for p in model_paths]
    if not paths:
        return {}
    if not (smiles or ccd):
        logger.info(
            "No ligand SMILES/CCD provided; skipping affinity scoring."
        )
        return {}

    affinity_dir = Path(output_dir) / "affinity"
    gpu_slots = detect_gpu_slots(gpus)
    use_sharding = (
        len(paths) >= 4
        and len(gpu_slots) > 1
        and gpu_slots != ["cpu"]
        and gpu_slots != ["all"]
    )

    def _run_shard(
        shard_paths: List[Path],
        shard_dir: Path,
        shard_gpu: Optional[str],
    ) -> Dict[str, dict]:
        cmd = [
            sys.executable,
            "-m",
            "abcfold.scripts.boltz_affinity",
            *[str(p) for p in shard_paths],
            "--output_dir",
            str(shard_dir),
        ]
        if smiles:
            cmd += ["--smiles", smiles]
        elif ccd:
            cmd += ["--ccd", ccd]
        if ligand_chain:
            cmd += ["--ligand_chain", ligand_chain]
        if msa:
            cmd += ["--msa", str(msa)]
        if shard_gpu:
            cmd += ["--gpus", str(shard_gpu)]
        if config and config.get("boltz_env"):
            cmd += ["--boltz_env", str(config["boltz_env"])]
        if config and config.get("boltz_weights"):
            cmd += ["--cache", str(config["boltz_weights"])]
        env_map = {**os.environ, **fast_torch_env()}
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, env=env_map)
        except FileNotFoundError as exc:
            logger.error("Could not launch affinity scoring: %s", exc)
            return {}
        if result.returncode != 0:
            tail = "\n".join((result.stderr or result.stdout or "").splitlines()[-25:])
            logger.error("Affinity scoring failed:\n%s", tail)
            return {}
        results_csv = shard_dir / "boltz_affinity_results.csv"
        if not results_csv.exists():
            logger.warning("No affinity results were produced at %s", results_csv)
            return {}
        df = pd.read_csv(results_csv)
        shard_scores: Dict[str, dict] = {}
        for _, row in df.iterrows():
            model = str(row.get("input_model", ""))
            if not model:
                continue
            key = str(Path(model).resolve())
            val = row.get("affinity_pred_value")
            prob = row.get("affinity_probability_binary")
            shard_scores[key] = {
                "affinity_pred_value": None if pd.isna(val) else float(val),
                "affinity_probability_binary": None if pd.isna(prob) else float(prob),
            }
        return shard_scores

    logger.info("Running Boltz-2 affinity scoring on %d model(s)...", len(paths))
    if not use_sharding:
        return _run_shard(paths, affinity_dir, gpus)

    n_shards = min(len(gpu_slots), len(paths))
    shards: List[List[Path]] = [[] for _ in range(n_shards)]
    for idx, p in enumerate(paths):
        shards[idx % n_shards].append(p)
    merged: Dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=n_shards) as pool:
        futs = [
            pool.submit(
                _run_shard, shards[i], affinity_dir / f"shard_{i}", gpu_slots[i]
            )
            for i in range(n_shards)
        ]
        for fut in as_completed(futs):
            merged.update(fut.result())
    return merged

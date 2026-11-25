"""Batch mode: run many co-folding jobs from a CSV/JSON manifest.

Manifest rows (CSV header or JSON object keys):

  id          job id (required, unique; used as the job name)
  protein     protein sequence(s); several chains separated by ':' or ';'
  af3_json    path to an AlphaFold 3 input JSON (instead of protein/ligands)
  smiles      ligand SMILES; several separated by ';'
  ccd         ligand/ion CCD codes; several separated by ';' (e.g. "MG;GDP")
  models      engines, separated by ';', ',' or ' ' (default: boltz)
  n_models    samples per engine (optional)
  seeds       model seeds separated by ';' (optional, default 1)

JSON manifests are a list of such objects or ``{"defaults": {...}, "jobs":
[...]}``; list values may be JSON arrays instead of separated strings.

MSAs for all unique protein sequences are fetched once before the jobs start
(``--msa server``) and shared through the MSA cache. Each job runs ``abcfold
run`` in its own process; ``batch_status.json`` records per-job state and a
re-run skips completed jobs.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import queue
import shlex
import shutil
import string
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

logger = logging.getLogger("logger")

STATUS_SCHEMA = "abcfold-batch-status-v1"
ENGINE_ALIASES = {
    "alphafold3": "alphafold3", "af3": "alphafold3", "alphafold": "alphafold3",
    "boltz": "boltz", "boltz2": "boltz", "boltz-2": "boltz",
    "chai": "chai1", "chai1": "chai1", "chai-1": "chai1",
    "protenix": "protenix",
    "openfold3": "openfold3", "openfold": "openfold3", "of3": "openfold3",
    "rosettafold3": "rosettafold3", "rf3": "rosettafold3",
    "intellifold": "intellifold", "if": "intellifold",
    "opendde": "opendde", "dde": "opendde",
    "esmfold2": "esmfold2", "esmfold": "esmfold2",
}
BATCHED_ENGINES = (
    "boltz",
    "chai1",
    "protenix",
    "openfold3",
    "opendde",
    "esmfold2",
)
ALL_ENGINES = ("alphafold3", "boltz", "chai1", "protenix", "openfold3",
               "rosettafold3")
DONE_STATES = ("ok",)


class ManifestError(ValueError):
    pass


@dataclass
class BatchJob:
    id: str
    params: dict
    models: List[str]
    n_models: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def protein_sequences(self) -> List[str]:
        return [
            s["protein"]["sequence"] for s in self.params.get("sequences", [])
            if "protein" in s
        ]


def _split(value: Any, seps: str = ";") -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value)
    for sep in seps[1:]:
        text = text.replace(sep, seps[0])
    return [part.strip() for part in text.split(seps[0]) if part.strip()]


def parse_models(value: Any) -> List[str]:
    names = _split(value, "; ,") or ["boltz"]
    if len(names) == 1 and names[0].lower() == "all":
        return list(ALL_ENGINES)
    models = []
    for name in names:
        key = ENGINE_ALIASES.get(name.lower())
        if key is None:
            raise ManifestError(f"Unknown model {name!r}")
        if key not in models:
            models.append(key)
    return models


def _chain_ids():
    letters = string.ascii_uppercase
    for letter in letters:
        yield letter
    for first in letters:
        for second in letters:
            yield first + second


def build_job_params(row: Mapping[str, Any], base_dir: Path) -> dict:
    job_id = str(row["id"])
    if row.get("af3_json"):
        path = Path(str(row["af3_json"])).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        params = json.loads(path.read_text())
        params["name"] = job_id
        return params

    proteins = _split(row.get("protein") or row.get("proteins"), ";:")
    smiles = _split(row.get("smiles") or row.get("ligand_smiles"))
    ccds = _split(row.get("ccd") or row.get("ligand_ccd"))
    if not proteins and not row.get("sequences"):
        raise ManifestError(f"Job {job_id}: needs 'protein' or 'af3_json'")
    ids = _chain_ids()
    sequences: List[dict] = []
    for seq in proteins:
        sequences.append({"protein": {"id": next(ids), "sequence": seq.upper()}})
    for smi in smiles:
        sequences.append({"ligand": {"id": next(ids), "smiles": smi}})
    for ccd in ccds:
        sequences.append({"ligand": {"id": next(ids), "ccdCodes": [ccd.upper()]}})
    seeds = [int(s) for s in _split(row.get("seeds"))] or [1]
    return {
        "name": job_id,
        "modelSeeds": seeds,
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 2,
    }


def _rows_from_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        reader = (
            csv.DictReader(
                handle, dialect=csv.Sniffer().sniff(sample, delimiters=",\t")
            )
            if sample
            else csv.DictReader(handle)
        )
        return [
            {k.strip().lower(): (v.strip() if isinstance(v, str) else v)
             for k, v in row.items() if k}
            for row in reader
            if any((v or "").strip() for v in row.values() if isinstance(v, str))
        ]


def _rows_from_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    defaults: Dict[str, Any] = {}
    if isinstance(data, dict):
        defaults = data.get("defaults", {})
        data = data.get("jobs", [])
    if not isinstance(data, list):
        raise ManifestError("JSON manifest must be a list or {'jobs': [...]} ")
    return [{**defaults, **{k.lower(): v for k, v in row.items()}} for row in data]


def parse_manifest(path: Path | str) -> List[BatchJob]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        rows = _rows_from_json(path)
    elif path.suffix.lower() in (".csv", ".tsv"):
        rows = _rows_from_csv(path)
    else:
        raise ManifestError(f"Manifest must be .csv, .tsv or .json: {path}")
    jobs: List[BatchJob] = []
    seen = set()
    for i, row in enumerate(rows, start=1):
        job_id = str(row.get("id") or "").strip()
        if not job_id:
            raise ManifestError(f"Row {i}: missing 'id'")
        if any(c in job_id for c in "/\\") or job_id in (".", ".."):
            raise ManifestError(f"Row {i}: invalid id {job_id!r}")
        if job_id in seen:
            raise ManifestError(f"Duplicate job id {job_id!r}")
        seen.add(job_id)
        n_models = row.get("n_models") or row.get("number_of_models")
        jobs.append(BatchJob(
            id=job_id,
            params=build_job_params({**row, "id": job_id}, path.parent),
            models=parse_models(row.get("models")),
            n_models=int(str(n_models)) if n_models not in (None, "") else None,
        ))
    if not jobs:
        raise ManifestError(f"No jobs in {path}")
    return jobs


class BatchStatus:
    """Thread-safe, atomically written ``batch_status.json``."""

    def __init__(self, path: Path, manifest: Optional[Path] = None):
        self.path = Path(path)
        self.lock = threading.Lock()
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {"schema": STATUS_SCHEMA, "manifest": None, "jobs": {}}
        if manifest is not None:
            self.data["manifest"] = str(manifest)

    def job(self, job_id: str) -> Dict[str, Any]:
        return self.data["jobs"].get(job_id, {})

    def is_done(self, job_id: str) -> bool:
        return self.job(job_id).get("status") in DONE_STATES

    def update(self, job_id: str, **fields: Any) -> None:
        with self.lock:
            entry = self.data["jobs"].setdefault(job_id, {})
            entry.update(fields)
            self.data["updated"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )
            counts: Dict[str, int] = {}
            for job in self.data["jobs"].values():
                counts[job.get("status", "pending")] = counts.get(
                    job.get("status", "pending"), 0) + 1
            self.data["counts"] = counts
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".status.")
            with os.fdopen(fd, "w") as handle:
                json.dump(self.data, handle, indent=2)
            os.replace(tmp, self.path)


def job_command(
    job: BatchJob,
    input_json: Path,
    output_dir: Path,
    gpu: Optional[str],
    msa: str,
    run_args: Sequence[str] = (),
) -> List[str]:
    cmd = [sys.executable, "-m", "abcfold.cli", "run", str(input_json),
           str(output_dir), "--override", "--no_visuals", "--msa", msa]
    cmd += [f"--{engine}" for engine in job.models]
    if job.n_models is not None:
        cmd += ["--number_of_models", str(job.n_models)]
    if gpu is not None:
        cmd += ["--gpus", gpu]
    return cmd + list(run_args)


def exit_status(code: int) -> str:
    return {0: "ok", 2: "partial"}.get(code, "failed")


def prefetch_msas(jobs: Sequence[BatchJob], options) -> Dict[str, int]:
    from abcfold.msa.resolve import prefetch_unique_sequences

    sequences = [s for job in jobs for s in job.protein_sequences]
    if not sequences:
        return {"fetched": 0, "cached": 0}
    fetched, cached = prefetch_unique_sequences(sequences, options)
    logger.info("MSA prefetch: %d fetched, %d already cached (%d unique)",
                fetched, cached, fetched + cached)
    return {"fetched": fetched, "cached": cached}


def estimate_job_tokens(job: BatchJob) -> int:
    """Estimate complex token length for AlphaFast-style length ordering."""
    tokens = 0
    for entry in job.params.get("sequences", []):
        for key in ("protein", "rna", "dna"):
            if key in entry:
                item = entry[key]
                chain_id = item.get("id", "A")
                copies = len(chain_id) if isinstance(chain_id, list) else 1
                tokens += len(item.get("sequence", "")) * max(1, copies)
        if "ligand" in entry:
            lig = entry["ligand"]
            chain_id = lig.get("id", "L")
            copies = len(chain_id) if isinstance(chain_id, list) else 1
            smiles = lig.get("smiles")
            if smiles:
                tokens += max(1, len(smiles) // 2) * max(1, copies)
            else:
                tokens += max(1, len(lig.get("ccdCodes", [])) * 8) * max(1, copies)
    return tokens


def sort_jobs_by_length(
    jobs: Sequence[BatchJob], descending: bool = True
) -> List[BatchJob]:
    """Order jobs by estimated token length (stable sort).

    Longest-first (LPT) minimizes multi-GPU makespan while grouping similar
    sequence lengths together to reduce XLA/CUDA allocator churn.
    """
    return sorted(jobs, key=estimate_job_tokens, reverse=descending)


_BATCHED_RUN_ARGS_SPEC: Dict[str, tuple[str, Any]] = {
    "--config-file": ("config_file", str),
    "--msa_cache_dir": ("msa_cache_dir", str),
    "--msa_server_url": ("msa_server_url", str),
    "--num_recycles": ("num_recycles", int),
    "--diffusion_steps": ("diffusion_steps", int),
    "--sampling_steps": ("diffusion_steps", int),
    "--number_of_models": ("number_of_models", int),
    "--max_msa_seqs": ("max_msa_seqs", int),
    "--min_msa_rows": ("min_msa_rows", int),
    "--boltz_preprocessing_threads": ("boltz_preprocessing_threads", int),
    "--boltz_num_workers": ("boltz_num_workers", int),
    "--boltz_max_parallel_samples": ("boltz_max_parallel_samples", int),
    "--protenix_model": ("protenix_model", str),
    "--template_mirror_dir": ("template_mirror_dir", str),
    "--inference_ckpt_path": ("inference_ckpt_path", str),
}
_BATCHED_BOOL_FLAGS: Dict[str, str] = {
    "--save_input": "save_input",
    "--boltz_use_potentials": "boltz_use_potentials",
    "--openfold_fetch_templates": "openfold_fetch_templates",
    "--allow_version_mismatch": "allow_version_mismatch",
    "--no_paired_msa": "no_paired_msa",
    "--no_visuals": "no_visuals",
    "--no_server": "no_server",
    "--override": "override",
}


def _parse_batch_run_args(run_args: Sequence[str]) -> Optional[Dict[str, Any]]:
    """Parse supported abcfold run flags for phase-separated batch execution.

    Returns None when an unrecognized or per-job specialized flag (such as
    --mmseqs2, --custom_template, --boltz_mode, --ligand_smiles) is present so
    run_batch falls back cleanly to per-job subprocess execution.
    """
    parsed: Dict[str, Any] = {
        "config_file": None,
        "msa_cache_dir": None,
        "msa_server_url": None,
        "num_recycles": 10,
        "diffusion_steps": None,
        "number_of_models": 5,
        "max_msa_seqs": None,
        "min_msa_rows": None,
        "boltz_preprocessing_threads": None,
        "boltz_num_workers": None,
        "boltz_max_parallel_samples": None,
        "protenix_model": None,
        "template_mirror_dir": None,
        "inference_ckpt_path": None,
        "save_input": False,
        "boltz_use_potentials": False,
        "openfold_fetch_templates": False,
        "allow_version_mismatch": False,
        "no_paired_msa": False,
    }
    i = 0
    args_list = list(run_args)
    while i < len(args_list):
        tok = args_list[i]
        if "=" in tok and tok.startswith("--"):
            flag, val = tok.split("=", 1)
            if flag not in _BATCHED_RUN_ARGS_SPEC:
                return None
            key, conv = _BATCHED_RUN_ARGS_SPEC[flag]
            try:
                parsed[key] = conv(val)
            except ValueError:
                return None
            i += 1
            continue
        if tok in _BATCHED_BOOL_FLAGS:
            parsed[_BATCHED_BOOL_FLAGS[tok]] = True
            i += 1
            continue
        if tok in _BATCHED_RUN_ARGS_SPEC:
            if i + 1 >= len(args_list):
                return None
            key, conv = _BATCHED_RUN_ARGS_SPEC[tok]
            try:
                parsed[key] = conv(args_list[i + 1])
            except ValueError:
                return None
            i += 2
            continue
        return None
    return parsed


def _run_batch_phased(
    todo: Sequence[BatchJob],
    output_dir: Path,
    inputs_dir: Path,
    logs_dir: Path,
    jobs_dir: Path,
    status: BatchStatus,
    gpus: Optional[str],
    msa: str,
    msa_options: Any,
    parsed_args: Dict[str, Any],
    run_args: Sequence[str],
    max_parallel: int,
) -> Dict[str, Any]:
    """AlphaFast phase-separated execution: MSA -> batched GPU -> parallel CPU.

    Loads each model's weights once per GPU across all pending jobs instead of
    spawning one cold Python + CUDA process per complex.
    """
    from abcfold.abcfold import (
        ENGINE_ORDER,
        EXIT_FAILED,
        EXIT_OK,
        EXIT_PARTIAL,
        abcfold_version,
    )
    from abcfold.msa.guard import (
        ShallowMsaError,
        check_input_msa_depth,
        min_msa_rows,
    )
    from abcfold.msa.resolve import MsaOptions, resolve_msas
    from abcfold.msa.stats import write_msa_stats
    from abcfold.output.boltz import BoltzOutput
    from abcfold.output.chai import ChaiOutput
    from abcfold.output.esmfold2 import Esmfold2Output
    from abcfold.output.opendde import OpenDDEOutput
    from abcfold.output.openfold3 import OpenfoldOutput
    from abcfold.output.protenix import ProtenixOutput
    from abcfold.report import (
        EngineRecord,
        build_report,
        samples_from_output,
        write_report,
    )
    from abcfold.scripts.abc_script_utils import limit_json_msa_depth
    from abcfold.settings import load_settings

    settings = load_settings(parsed_args["config_file"])
    overrides = {
        k: parsed_args[k]
        for k in (
            "msa_cache_dir",
            "msa_server_url",
            "boltz_preprocessing_threads",
            "boltz_num_workers",
            "boltz_max_parallel_samples",
            "template_mirror_dir",
            "min_msa_rows",
        )
        if parsed_args.get(k) is not None
    }
    settings = settings.apply_overrides(overrides)
    if parsed_args.get("protenix_model"):
        settings.backends["protenix"]["model"] = parsed_args["protenix_model"]
    if parsed_args.get("allow_version_mismatch"):
        settings.values["allow_version_mismatch"] = True
    rt_config = settings.runtime_config()

    effective_min_rows = min_msa_rows(rt_config)
    max_msa_seqs = parsed_args.get("max_msa_seqs")
    if max_msa_seqs and effective_min_rows > max_msa_seqs:
        effective_min_rows = max_msa_seqs
        rt_config["min_msa_rows"] = effective_min_rows

    if msa_options is None:
        msa_options = MsaOptions(
            source=msa,
            cache_dir=settings.get("msa_cache_dir"),
            server_url=settings.get("msa_server_url", "https://api.colabfold.com"),
            user_agent=settings.get("msa_user_agent", "abcfold"),
            paired=not parsed_args.get("no_paired_msa", False),
        )
    else:
        msa_options = MsaOptions(
            source=msa,
            cache_dir=msa_options.cache_dir or settings.get("msa_cache_dir"),
            server_url=msa_options.server_url or settings.get(
                "msa_server_url", "https://api.colabfold.com"
            ),
            user_agent=msa_options.user_agent,
            paired=not parsed_args.get("no_paired_msa", False),
        )

    job_started: Dict[str, float] = {}
    job_run_json: Dict[str, Path] = {}
    job_params: Dict[str, dict] = {}
    job_msa_section: Dict[str, Dict[str, Any]] = {}
    job_precheck_err: Dict[str, Optional[str]] = {}
    job_engine_ok: Dict[str, Dict[str, bool]] = {j.id: {} for j in todo}
    job_engine_sec: Dict[str, Dict[str, float]] = {j.id: {} for j in todo}
    job_engine_err: Dict[str, Dict[str, str]] = {j.id: {} for j in todo}

    def _prepare_job(job: BatchJob) -> None:
        job_started[job.id] = time.monotonic()
        input_json = inputs_dir / f"{job.id}.json"
        input_json.write_text(json.dumps(job.params, indent=2))
        job_out = jobs_dir / job.id
        job_out.mkdir(parents=True, exist_ok=True)
        status.update(
            job.id,
            status="running",
            gpu=gpus,
            started=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            output_dir=str(job_out),
            log=str(logs_dir / f"{job.id}.log"),
            mode="batched",
        )
        res = resolve_msas(job.params, msa_options)
        params = res.params
        section = {
            "source": res.source,
            "seconds": round(res.seconds, 3),
            "chains": res.chains,
        }
        if res.chains:
            write_msa_stats(
                job_out / "msa_stats.json", res.chains, res.source, res.seconds
            )
        run_json = job_out / f"abc_{job.id}.json"
        run_json.write_text(json.dumps(params, indent=2))
        if max_msa_seqs:
            params = limit_json_msa_depth(run_json, run_json, max_msa_seqs)
        pre_err: Optional[str] = None
        try:
            check_input_msa_depth(params, effective_min_rows, context=run_json.name)
        except ShallowMsaError as exc:
            pre_err = str(exc)
        job_run_json[job.id] = run_json
        job_params[job.id] = params
        job_msa_section[job.id] = section
        job_precheck_err[job.id] = pre_err

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(todo)))) as pool:
        list(pool.map(_prepare_job, todo))

    valid_jobs = [j for j in todo if job_precheck_err[j.id] is None]
    gpu_str = gpus or "all"
    save_input = bool(parsed_args.get("save_input", False))
    num_recycles = int(parsed_args.get("num_recycles", 10))
    diffusion_steps = parsed_args.get("diffusion_steps")
    default_n_models = int(parsed_args.get("number_of_models", 5))

    batch_stage = output_dir / "_batched_stage"
    batch_stage.mkdir(parents=True, exist_ok=True)
    fallback_by_job: Dict[str, List[str]] = {}

    for engine in ENGINE_ORDER:
        eng_jobs = [j for j in valid_jobs if engine in j.models]
        if not eng_jobs:
            continue
        if engine not in BATCHED_ENGINES:
            for j in eng_jobs:
                fallback_by_job.setdefault(j.id, []).append(engine)
            continue

        by_n_models: Dict[int, List[BatchJob]] = {}
        for j in eng_jobs:
            n_mod = j.n_models if j.n_models is not None else default_n_models
            by_n_models.setdefault(n_mod, []).append(j)

        for n_mod, group in by_n_models.items():
            t0 = time.monotonic()
            logger.info(
                "Running batched %s for %d job(s) (n_models=%d, gpus=%s)",
                engine,
                len(group),
                n_mod,
                gpu_str,
            )
            if engine == "boltz":
                from abcfold.boltz.run_boltz import run_boltz_batch

                stage_dir = batch_stage / f"boltz_n{n_mod}"
                ok = run_boltz_batch(
                    [job_run_json[j.id] for j in group],
                    stage_dir,
                    config=rt_config,
                    save_input=save_input,
                    number_of_models=n_mod,
                    num_recycles=num_recycles,
                    gpus=gpu_str,
                    preprocessing_threads=settings.get("boltz_preprocessing_threads"),
                    num_workers=settings.get("boltz_num_workers"),
                    max_parallel_samples=settings.get("boltz_max_parallel_samples"),
                    sampling_steps=diffusion_steps or 200,
                    boltz_use_potentials=bool(
                        parsed_args.get("boltz_use_potentials", False)
                    ),
                )
                elapsed = time.monotonic() - t0
                per_job_sec = elapsed / len(group)
                for j in group:
                    job_engine_sec[j.id][engine] = per_job_sec
                    stem = job_run_json[j.id].stem
                    job_out = jobs_dir / j.id
                    seeds_raw = job_params[j.id].get("modelSeeds", [1])
                    seeds_list = (
                        [int(seeds_raw)]
                        if isinstance(seeds_raw, int)
                        else [int(s) for s in seeds_raw]
                    )
                    moved_any = False
                    if ok:
                        for seed in seeds_list:
                            src_pred = (
                                stage_dir
                                / "boltz_batch"
                                / f"seed-{seed}"
                                / f"boltz_results_seed-{seed}"
                                / "predictions"
                                / stem
                            )
                            if src_pred.exists() and list(
                                src_pred.glob("*_model_*.cif")
                            ):
                                dst_pred = (
                                    job_out
                                    / f"boltz_results_{stem}_seed-{seed}"
                                    / "predictions"
                                    / stem
                                )
                                if dst_pred.exists():
                                    shutil.rmtree(dst_pred)
                                dst_pred.parent.mkdir(parents=True, exist_ok=True)
                                shutil.move(str(src_pred), str(dst_pred))
                                moved_any = True
                    job_engine_ok[j.id][engine] = moved_any
                    if not moved_any:
                        job_engine_err[j.id][engine] = (
                            "Boltz batch produced no structures"
                        )

            elif engine == "chai1":
                from abcfold.chai1.run_chai1_fast import run_chai_batch

                stage_dir = batch_stage / f"chai1_n{n_mod}"
                ok = run_chai_batch(
                    [job_run_json[j.id] for j in group],
                    stage_dir,
                    config=rt_config,
                    save_input=save_input,
                    number_of_models=n_mod,
                    num_recycles=num_recycles,
                    devices=gpu_str,
                    nested_outputs=True,
                    postprocess=False,
                    diffusion_steps=diffusion_steps or 200,
                )
                elapsed = time.monotonic() - t0
                per_job_sec = elapsed / len(group)
                for j in group:
                    job_engine_sec[j.id][engine] = per_job_sec
                    case_id = job_params[j.id].get("name") or job_run_json[j.id].stem
                    src_case = stage_dir / "outputs" / case_id
                    job_out = jobs_dir / j.id
                    moved_any = False
                    if ok and src_case.exists():
                        for src_dir in src_case.glob("chai_output*"):
                            if list(src_dir.glob("pred.model_idx_*.cif")):
                                dst_dir = job_out / src_dir.name
                                if dst_dir.exists():
                                    shutil.rmtree(dst_dir)
                                shutil.move(str(src_dir), str(dst_dir))
                                moved_any = True
                    job_engine_ok[j.id][engine] = moved_any
                    if not moved_any:
                        job_engine_err[j.id][engine] = (
                            "Chai-1 batch produced no structures"
                        )

            elif engine == "protenix":
                from abcfold.protenix.run_protenix import run_protenix_batch

                res_map = run_protenix_batch(
                    [(job_run_json[j.id], jobs_dir / j.id) for j in group],
                    config=rt_config,
                    save_input=save_input,
                    number_of_models=n_mod,
                    num_recycles=num_recycles,
                    diffusion_steps=diffusion_steps,
                    gpus=gpu_str,
                )
                elapsed = time.monotonic() - t0
                per_job_sec = elapsed / len(group)
                for j in group:
                    job_engine_sec[j.id][engine] = per_job_sec
                    job_engine_ok[j.id][engine] = bool(res_map.get(jobs_dir / j.id))
                    if not job_engine_ok[j.id][engine]:
                        job_engine_err[j.id][engine] = (
                            "Protenix batch produced no structures"
                        )

            elif engine == "openfold3":
                from abcfold.openfold3.run_openfold3 import run_openfold_batch

                res_map = run_openfold_batch(
                    [(job_run_json[j.id], jobs_dir / j.id) for j in group],
                    config=rt_config,
                    save_input=save_input,
                    number_of_models=n_mod,
                    input_ckpt=parsed_args.get("inference_ckpt_path"),
                    num_recycles=num_recycles,
                    diffusion_steps=diffusion_steps,
                    fetch_templates=bool(
                        parsed_args.get("openfold_fetch_templates", False)
                    ),
                    template_mirror_dir=settings.get("template_mirror_dir"),
                    gpus=gpu_str,
                )
                elapsed = time.monotonic() - t0
                per_job_sec = elapsed / len(group)
                for j in group:
                    job_engine_sec[j.id][engine] = per_job_sec
                    job_engine_ok[j.id][engine] = bool(res_map.get(jobs_dir / j.id))
                    if not job_engine_ok[j.id][engine]:
                        job_engine_err[j.id][engine] = (
                            "OpenFold3 batch produced no structures"
                        )

            elif engine == "opendde":
                from abcfold.opendde.run_opendde import run_opendde_batch

                res_map = run_opendde_batch(
                    [(job_run_json[j.id], jobs_dir / j.id) for j in group],
                    config=rt_config,
                    save_input=save_input,
                    number_of_models=n_mod,
                    num_recycles=num_recycles,
                    gpus=gpu_str,
                )
                elapsed = time.monotonic() - t0
                per_job_sec = elapsed / len(group)
                for j in group:
                    job_engine_sec[j.id][engine] = per_job_sec
                    job_engine_ok[j.id][engine] = bool(res_map.get(jobs_dir / j.id))
                    if not job_engine_ok[j.id][engine]:
                        job_engine_err[j.id][engine] = (
                            "OpenDDE batch produced no structures"
                        )

            elif engine == "esmfold2":
                from abcfold.esmfold2.run_esmfold2 import run_esmfold2_batch

                res_map = run_esmfold2_batch(
                    [(job_run_json[j.id], jobs_dir / j.id) for j in group],
                    config=rt_config,
                    save_input=save_input,
                    number_of_models=n_mod,
                    gpus=gpu_str,
                )
                elapsed = time.monotonic() - t0
                per_job_sec = elapsed / len(group)
                for j in group:
                    job_engine_sec[j.id][engine] = per_job_sec
                    job_engine_ok[j.id][engine] = bool(res_map.get(jobs_dir / j.id))
                    if not job_engine_ok[j.id][engine]:
                        job_engine_err[j.id][engine] = (
                            "ESMFold2 batch produced no structures"
                        )

    if fallback_by_job:
        gpu_list = _split(gpus, ",") if gpus and gpus.lower() not in ("all",) else []
        slots: "queue.Queue[Optional[str]]" = queue.Queue()
        for i in range(max(1, max_parallel, len(gpu_list))):
            slots.put(gpu_list[i % len(gpu_list)] if gpu_list else gpus)

        def _run_fallback(job_id: str, engines_list: List[str]) -> None:
            gpu = slots.get()
            try:
                job_obj = next(j for j in todo if j.id == job_id)
                sub_job = BatchJob(
                    id=job_obj.id,
                    params=job_obj.params,
                    models=engines_list,
                    n_models=job_obj.n_models,
                )
                job_out = jobs_dir / job_id
                log_path = logs_dir / f"{job_id}_extra.log"
                cmd = job_command(
                    sub_job, job_run_json[job_id], job_out, gpu, "auto", run_args
                )
                t0 = time.monotonic()
                code = _subprocess_runner(cmd, log_path, None)
                elapsed = time.monotonic() - t0
                for eng in engines_list:
                    job_engine_sec[job_id][eng] = elapsed / len(engines_list)
                    job_engine_ok[job_id][eng] = code == 0
                    if code != 0:
                        job_engine_err[job_id][eng] = f"subprocess exited with {code}"
            finally:
                slots.put(gpu)

        with ThreadPoolExecutor(
            max_workers=max(1, max_parallel, len(gpu_list))
        ) as pool:
            futs = [
                pool.submit(_run_fallback, jid, engs)
                for jid, engs in fallback_by_job.items()
            ]
            for fut in futs:
                fut.result()

    def _finalize_job(job: BatchJob) -> None:
        job_out = jobs_dir / job.id
        params = job_params[job.id]
        name = params.get("name") or job.id
        records = {e: EngineRecord(engine=e) for e in ENGINE_ORDER}
        pre_err = job_precheck_err[job.id]

        for engine in job.models:
            rec = records[engine]
            rec.version = settings.backend(engine).get("version")
            rec.seconds = job_engine_sec[job.id].get(engine, 0.0)
            if pre_err is not None:
                rec.status = "failed"
                rec.error = pre_err
                continue
            if not job_engine_ok[job.id].get(engine, False):
                rec.status = "failed"
                rec.error = job_engine_err[job.id].get(
                    engine, "backend produced no structures"
                )
                continue
            rec.status = "ok"
            out_obj: Any = None
            try:
                dirs: List[Union[str, Path]]
                if engine == "boltz":
                    dirs = list(job_out.glob("boltz_results*"))
                    if dirs:
                        out_obj = BoltzOutput(dirs, params, name, save_input)
                elif engine == "chai1":
                    dirs = list(job_out.glob("chai_output*"))
                    if dirs:
                        out_obj = ChaiOutput(dirs, params, name, rt_config, save_input)
                elif engine == "protenix":
                    dirs = list(job_out.glob("protenix_results*"))
                    if dirs:
                        out_obj = ProtenixOutput(dirs, params, name, save_input)
                elif engine == "openfold3":
                    dirs = list(job_out.glob("openfold_results*"))
                    if dirs:
                        out_obj = OpenfoldOutput(dirs, params, name, save_input)
                elif engine == "opendde":
                    dirs = list(job_out.glob("opendde_results*"))
                    if dirs:
                        out_obj = OpenDDEOutput(dirs, params, name, save_input)
                elif engine == "esmfold2":
                    dirs = list(job_out.glob("esmfold2_results*"))
                    if dirs:
                        out_obj = Esmfold2Output(dirs, params, name, save_input)
                if out_obj is not None:
                    n = sum(
                        1
                        for models in out_obj.output.values()
                        for idx in models
                        if not (isinstance(idx, int) and idx < 0)
                    )
                    rec.samples = samples_from_output(
                        engine,
                        rec.version,
                        out_obj.output,
                        job_out,
                        seconds_per_sample=(rec.seconds / n) if n else None,
                    )
            except Exception as exc:
                logger.warning("Could not summarise %s for %s: %s", engine, job.id, exc)

        report = build_report(
            name=name,
            engines=[records[e] for e in ENGINE_ORDER],
            msa=job_msa_section[job.id],
            input_json=str(job_run_json[job.id]),
            output_dir=str(job_out.resolve()),
            abcfold_version=abcfold_version(),
        )
        report_path = write_report(report, job_out / "report.json")
        ok_engines = [e for e in job.models if records[e].status == "ok"]
        code = (
            EXIT_OK
            if len(ok_engines) == len(job.models)
            else EXIT_PARTIAL
            if ok_engines
            else EXIT_FAILED
        )
        status.update(
            job.id,
            status=exit_status(code),
            exit_code=code,
            seconds=round(time.monotonic() - job_started[job.id], 2),
            finished=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            report=str(report_path),
            engines={
                e["engine"]: e["status"]
                for e in report.get("engines", [])
                if e.get("status") != "skipped"
            },
        )
        logger.info("Job %s: %s", job.id, exit_status(code))

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(todo)))) as pool:
        list(pool.map(_finalize_job, todo))

    shutil.rmtree(batch_stage, ignore_errors=True)
    return status.data


def run_batch(
    manifest: Path | str,
    output_dir: Path | str,
    max_parallel: int = 1,
    gpus: Optional[str] = None,
    msa: str = "server",
    msa_options=None,
    run_args: Sequence[str] = (),
    force: bool = False,
    runner=None,
    batched: bool = True,
) -> Dict[str, Any]:
    """Run every job of manifest; returns the final status mapping.

    runner(cmd, log_path, env) -> int is injectable for tests.
    When runner is None and batched=True with >1 pending job, runs AlphaFast-style
    phase-separated batching (MSA prefetch -> batched GPU weights-loaded-once ->
    parallel CPU post-processing).
    """
    manifest = Path(manifest)
    output_dir = Path(output_dir)
    jobs = parse_manifest(manifest)
    status = BatchStatus(output_dir / "batch_status.json", manifest)
    inputs_dir = output_dir / "inputs"
    logs_dir = output_dir / "logs"
    jobs_dir = output_dir / "jobs"
    for d in (inputs_dir, logs_dir, jobs_dir):
        d.mkdir(parents=True, exist_ok=True)

    todo = [j for j in jobs if force or not status.is_done(j.id)]
    for job in jobs:
        if job not in todo:
            logger.info("Skipping completed job %s", job.id)
        elif status.job(job.id).get("status") != "pending":
            status.update(job.id, status="pending", models=job.models)

    if len(todo) > 1:
        todo = sort_jobs_by_length(todo, descending=True)

    if msa == "server" and todo:
        if msa_options is None:
            from abcfold.msa.resolve import MsaOptions

            msa_options = MsaOptions(source="server")
        try:
            prefetch_msas(todo, msa_options)
        except Exception as exc:  # jobs will retry per complex
            logger.error("MSA prefetch failed: %s", exc)

    if runner is None and batched and len(todo) > 1:
        parsed = _parse_batch_run_args(run_args)
        if parsed is not None:
            return _run_batch_phased(
                todo=todo,
                output_dir=output_dir,
                inputs_dir=inputs_dir,
                logs_dir=logs_dir,
                jobs_dir=jobs_dir,
                status=status,
                gpus=gpus,
                msa=msa,
                msa_options=msa_options,
                parsed_args=parsed,
                run_args=run_args,
                max_parallel=max_parallel,
            )

    gpu_list = _split(gpus, ",") if gpus and gpus.lower() not in ("all",) else []
    slots: "queue.Queue[Optional[str]]" = queue.Queue()
    for i in range(max(1, max_parallel)):
        slots.put(gpu_list[i % len(gpu_list)] if gpu_list else gpus)

    runner = runner or _subprocess_runner

    def execute(job: BatchJob) -> None:
        gpu = slots.get()
        try:
            input_json = inputs_dir / f"{job.id}.json"
            input_json.write_text(json.dumps(job.params, indent=2))
            job_out = jobs_dir / job.id
            log_path = logs_dir / f"{job.id}.log"
            cmd = job_command(job, input_json, job_out, gpu, msa, run_args)
            started = time.monotonic()
            status.update(job.id, status="running", gpu=gpu,
                          started=datetime.now(timezone.utc).isoformat(
                              timespec="seconds"),
                          output_dir=str(job_out), log=str(log_path),
                          command=shlex.join(cmd))
            try:
                code = runner(cmd, log_path, None)
            except Exception as exc:
                code = 1
                log_path.write_text(f"{type(exc).__name__}: {exc}\n")
            report = job_out / "report.json"
            fields: Dict[str, Any] = {
                "status": exit_status(code),
                "exit_code": code,
                "seconds": round(time.monotonic() - started, 2),
                "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "report": str(report) if report.exists() else None,
            }
            if report.exists():
                try:
                    data = json.loads(report.read_text())
                    fields["engines"] = {
                        e["engine"]: e["status"] for e in data.get("engines", [])
                        if e.get("status") != "skipped"
                    }
                except (OSError, ValueError):
                    pass
            status.update(job.id, **fields)
            logger.info("Job %s: %s (%.0fs)", job.id, fields["status"],
                        fields["seconds"])
        finally:
            slots.put(gpu)

    if max_parallel <= 1:
        for job in todo:
            execute(job)
    else:
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            list(pool.map(execute, todo))
    return status.data


def _subprocess_runner(cmd: List[str], log_path: Path, env) -> int:
    env = dict(os.environ if env is None else env)
    env.pop("LD_PRELOAD", None)
    with log_path.open("w") as log:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                              env=env, check=False).returncode

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
from typing import Any, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger("logger")

STATUS_SCHEMA = "abcfold-batch-status-v1"
ENGINE_ALIASES = {
    "alphafold3": "alphafold3", "af3": "alphafold3", "alphafold": "alphafold3",
    "boltz": "boltz", "boltz2": "boltz", "boltz-2": "boltz",
    "chai": "chai1", "chai1": "chai1", "chai-1": "chai1",
    "protenix": "protenix",
    "openfold3": "openfold3", "openfold": "openfold3", "of3": "openfold3",
    "rosettafold3": "rosettafold3", "rf3": "rosettafold3",
}
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
) -> Dict[str, Any]:
    """Run every job of ``manifest``; returns the final status mapping.

    ``runner(cmd, log_path, env) -> int`` is injectable for tests.
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

    if msa == "server" and todo:
        if msa_options is None:
            from abcfold.msa.resolve import MsaOptions

            msa_options = MsaOptions(source="server")
        try:
            prefetch_msas(todo, msa_options)
        except Exception as exc:  # jobs will retry per complex
            logger.error("MSA prefetch failed: %s", exc)

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

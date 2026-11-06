"""AlphaFold 3 adapter.

Runs the official AlphaFold 3 ``run_alphafold.py`` through Docker, Apptainer /
Singularity, or an arbitrary site wrapper command, and parses its output
directory.  The data pipeline (MSA/template search) and the inference stage can
be run separately: ``data_pipeline`` writes ``<name>/<name>_data.json``, which
``inference`` consumes without touching the databases.
"""

from __future__ import annotations

import copy
import csv
import json
import logging
import os
import shlex
import string
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from abcfold.gpus import docker_gpu_flags, to_physical

logger = logging.getLogger("logger")

RUNNERS = ("docker", "apptainer", "singularity", "command")
MODES = ("full", "data_pipeline", "inference")

CONTAINER_INPUT_DIR = "/root/af_input"
CONTAINER_OUTPUT_DIR = "/root/af_output"
CONTAINER_MODEL_DIR = "/root/models"
CONTAINER_DB_DIR = "/root/public_databases"
CONTAINER_APP_DIR = "/app/alphafold"


def sanitised_name(name: str) -> str:
    """Output directory name used by AlphaFold 3 (lower case, [a-z0-9_-.])."""
    lower_spaceless = name.lower().replace(" ", "_")
    allowed = set(string.ascii_lowercase + string.digits + "_-.")
    return "".join(c for c in lower_spaceless if c in allowed)


@dataclass
class Af3Config:
    runner: str = "docker"
    image: str = "alphafold3"
    model_dir: Optional[str] = None
    db_dir: Optional[str] = None
    command: Optional[str] = None
    append_flags: bool = True
    run_script: str = "run_alphafold.py"
    gpus: str = "all"
    num_diffusion_samples: int = 5
    num_recycles: Optional[int] = None
    num_seeds: Optional[int] = None
    save_distogram: bool = False
    extra_flags: List[str] = field(default_factory=list)

    @classmethod
    def from_runtime_config(cls, config: Mapping, **overrides) -> "Af3Config":
        def pick(key, default=None):
            value = config.get(key)
            return default if value in (None, "", "None") else value

        cfg = cls(
            runner=str(pick("af3_runner", "docker")),
            image=str(pick("af3_image", "alphafold3")),
            model_dir=pick("model_params"),
            db_dir=pick("database_dir"),
            command=pick("af3_command"),
            append_flags=str(pick("af3_append_flags", True)).lower()
            not in ("false", "0", "no"),
            run_script=str(pick("af3_run_script", "run_alphafold.py")),
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        return cfg


def _gpu_docker_flags(gpus: str) -> List[str]:
    return docker_gpu_flags(gpus)


def af3_flags(
    cfg: Af3Config,
    mode: str,
    json_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    model_dir: Optional[str] = None,
    db_dir: Optional[str] = None,
) -> List[str]:
    """Flags for ``run_alphafold.py``; path flags only when paths are given."""
    if mode not in MODES:
        raise ValueError(f"Unknown AF3 mode {mode!r}, expected one of {MODES}")
    flags: List[str] = []
    if json_path:
        flags.append(f"--json_path={json_path}")
    if model_dir and mode != "data_pipeline":
        flags.append(f"--model_dir={model_dir}")
    if db_dir and mode != "inference":
        flags.append(f"--db_dir={db_dir}")
    if output_dir:
        flags.append(f"--output_dir={output_dir}")
    if mode == "data_pipeline":
        flags.append("--run_inference=false")
    elif mode == "inference":
        flags.append("--run_data_pipeline=false")
    if mode != "data_pipeline":
        flags.append(f"--num_diffusion_samples={int(cfg.num_diffusion_samples)}")
        if cfg.num_recycles is not None:
            flags.append(f"--num_recycles={int(cfg.num_recycles)}")
        if cfg.num_seeds is not None:
            flags.append(f"--num_seeds={int(cfg.num_seeds)}")
        if cfg.save_distogram:
            flags.append("--save_distogram=true")
    flags.extend(cfg.extra_flags)
    return flags


def build_af3_command(
    input_json: Path | str,
    output_dir: Path | str,
    cfg: Af3Config,
    mode: str = "full",
) -> List[str]:
    """Build the argv for one AlphaFold 3 invocation (no shell)."""
    input_json = Path(input_json).resolve()
    output_dir = Path(output_dir).resolve()
    needs_models = mode != "data_pipeline"
    needs_dbs = mode != "inference"

    if cfg.runner == "command":
        if not cfg.command:
            raise ValueError("af3_runner=command needs af3_command")
        placeholders = {
            "input_json": str(input_json),
            "input_dir": str(input_json.parent),
            "json_name": input_json.name,
            "output_dir": str(output_dir),
            "model_dir": str(cfg.model_dir or ""),
            "db_dir": str(cfg.db_dir or ""),
            "gpus": "" if str(cfg.gpus) == "all" else str(cfg.gpus),
        }
        argv = [
            os.path.expanduser(part.format(**placeholders))
            for part in shlex.split(cfg.command)
        ]
        argv = [a for a in argv if a != ""]
        if cfg.append_flags:
            # Paths are the wrapper's business; only pass stage/sampling flags.
            argv += af3_flags(cfg, mode)
        return argv

    if needs_models and not cfg.model_dir:
        raise ValueError("AlphaFold 3 inference needs a model directory (--model_params)")
    if needs_dbs and not cfg.db_dir:
        raise ValueError(
            "The AlphaFold 3 data pipeline needs a database directory (--database)"
        )

    mounts = [
        (str(input_json.parent), CONTAINER_INPUT_DIR, True),
        (str(output_dir), CONTAINER_OUTPUT_DIR, False),
    ]
    if needs_models:
        mounts.append((str(Path(str(cfg.model_dir)).expanduser().resolve()),
                       CONTAINER_MODEL_DIR, True))
    if needs_dbs:
        mounts.append((str(Path(str(cfg.db_dir)).expanduser().resolve()),
                       CONTAINER_DB_DIR, True))

    flags = af3_flags(
        cfg,
        mode,
        json_path=f"{CONTAINER_INPUT_DIR}/{input_json.name}",
        output_dir=CONTAINER_OUTPUT_DIR,
        model_dir=CONTAINER_MODEL_DIR if needs_models else None,
        db_dir=CONTAINER_DB_DIR if needs_dbs else None,
    )

    if cfg.runner == "docker":
        argv = ["docker", "run", "--rm"]
        if mode != "data_pipeline":
            argv += _gpu_docker_flags(cfg.gpus)
        for src, dst, ro in mounts:
            argv += ["--volume", f"{src}:{dst}" + (":ro" if ro else "")]
        argv += [cfg.image, "python", cfg.run_script]
        return argv + flags

    if cfg.runner in ("apptainer", "singularity"):
        script = cfg.run_script
        if not script.startswith("/"):
            script = f"{CONTAINER_APP_DIR}/{script}"
        argv = [cfg.runner, "exec"]
        if mode != "data_pipeline" and str(cfg.gpus).lower() != "cpu":
            argv.append("--nv")
        for src, dst, ro in mounts:
            argv += ["--bind", f"{src}:{dst}" + (":ro" if ro else "")]
        argv += [str(Path(cfg.image).expanduser()), "python", script]
        return argv + flags

    raise ValueError(f"Unknown af3_runner {cfg.runner!r}, expected one of {RUNNERS}")


def af3_subprocess_env(cfg: Af3Config, environ: Optional[Mapping] = None) -> Dict:
    env = dict(os.environ if environ is None else environ)
    env.pop("LD_PRELOAD", None)
    gpus = str(cfg.gpus)
    if cfg.runner in ("apptainer", "singularity", "command") and gpus.lower() not in (
        "all", "cpu"
    ):
        physical = to_physical(gpus, environ=env)
        env["CUDA_VISIBLE_DEVICES"] = physical
        env["APPTAINERENV_CUDA_VISIBLE_DEVICES"] = physical
        env["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] = physical
    return env


def prepare_inference_input(params: dict) -> dict:
    """Make an input valid for ``--run_data_pipeline=false``.

    AlphaFold 3 refuses protein/RNA chains without explicit MSA fields when the
    data pipeline is skipped; missing fields become empty (single sequence).
    """
    params = copy.deepcopy(params)
    for entry in params.get("sequences", []):
        if "protein" in entry:
            protein = entry["protein"]
            if protein.get("unpairedMsa") is None and not protein.get("unpairedMsaPath"):
                logger.warning(
                    "AF3 inference-only: chain %s has no MSA, using single sequence",
                    protein.get("id"),
                )
                protein["unpairedMsa"] = ""
            if protein.get("pairedMsa") is None and not protein.get("pairedMsaPath"):
                protein["pairedMsa"] = ""
            if protein.get("templates") is None:
                protein["templates"] = []
        elif "rna" in entry:
            rna = entry["rna"]
            if rna.get("unpairedMsa") is None and not rna.get("unpairedMsaPath"):
                rna["unpairedMsa"] = ""
    return params


@dataclass
class Af3Sample:
    seed: int
    sample: int
    cif: Path
    summary: Optional[Path]
    confidences: Optional[Path]
    ranking_score: Optional[float] = None


@dataclass
class Af3Result:
    job_dir: Path
    name: str
    data_json: Optional[Path]
    top_model: Optional[Path]
    top_summary: Optional[Path]
    top_confidences: Optional[Path]
    samples: List[Af3Sample]

    @property
    def ok(self) -> bool:
        return self.top_model is not None or bool(self.samples)


def find_af3_job_dir(output_dir: Path | str, name: str) -> Optional[Path]:
    """Locate ``<output_dir>/<name>`` or AF3's timestamped ``<name>_YYYYMMDD_HHMMSS``."""
    output_dir = Path(output_dir)
    base = sanitised_name(name)
    candidates = [p for p in output_dir.glob(f"{base}*") if p.is_dir() and (
        p.name == base or p.name[len(base):].lstrip("_").replace("_", "").isdigit()
    )]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _first(paths: Sequence[Path]) -> Optional[Path]:
    return sorted(paths)[0] if paths else None


def _read_ranking(job_dir: Path) -> Dict[tuple, float]:
    scores: Dict[tuple, float] = {}
    for csv_path in job_dir.glob("*ranking_scores.csv"):
        with csv_path.open() as handle:
            for row in csv.DictReader(handle):
                try:
                    scores[(int(row["seed"]), int(row["sample"]))] = float(
                        row["ranking_score"]
                    )
                except (KeyError, ValueError):
                    continue
    return scores


def parse_af3_output(output_dir: Path | str, name: str) -> Optional[Af3Result]:
    job_dir = find_af3_job_dir(output_dir, name)
    if job_dir is None:
        return None
    ranking = _read_ranking(job_dir)
    samples: List[Af3Sample] = []
    for sample_dir in sorted(job_dir.glob("seed-*_sample-*")):
        if not sample_dir.is_dir():
            continue
        try:
            seed_part, sample_part = sample_dir.name.split("_sample-")
            seed = int(seed_part.split("seed-")[1])
            sample = int(sample_part)
        except (IndexError, ValueError):
            continue
        cif = _first(list(sample_dir.glob("*model.cif")))
        if cif is None:
            continue
        summary = _first(list(sample_dir.glob("*summary_confidences.json")))
        confidences = _first([
            p for p in sample_dir.glob("*confidences.json")
            if "summary" not in p.name
        ])
        score = ranking.get((seed, sample))
        if score is None and summary is not None:
            score = json.loads(summary.read_text()).get("ranking_score")
        samples.append(Af3Sample(seed, sample, cif, summary, confidences, score))

    top_model = _first([p for p in job_dir.glob("*_model.cif")])
    top_summary = _first(list(job_dir.glob("*_summary_confidences.json")))
    top_conf = _first([
        p for p in job_dir.glob("*_confidences.json") if "summary" not in p.name
    ])
    data_json = _first(list(job_dir.glob("*_data.json")))
    return Af3Result(job_dir, name, data_json, top_model, top_summary, top_conf,
                     samples)


def run_af3(
    input_params: dict,
    output_dir: Path | str,
    cfg: Af3Config,
    mode: str = "full",
    log_file: Optional[Path] = None,
) -> Af3Result:
    """Run AlphaFold 3 for ``input_params`` and return the parsed result.

    Raises ``RuntimeError`` if the process fails or produces no output.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = input_params.get("name")
    if not name:
        raise ValueError("Input JSON must contain a non-empty 'name' field")

    params = prepare_inference_input(input_params) if mode == "inference" else \
        input_params
    input_dir = output_dir / "af3_input"
    input_dir.mkdir(exist_ok=True)
    json_path = input_dir / f"{sanitised_name(name) or 'input'}.json"
    json_path.write_text(json.dumps(params, indent=2))

    argv = build_af3_command(json_path, output_dir, cfg, mode)
    logger.info("Running AlphaFold 3 (%s): %s", mode, shlex.join(argv))
    log_file = log_file or output_dir / "af3.log"
    start = time.monotonic()
    with log_file.open("w") as log:
        proc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                              env=af3_subprocess_env(cfg), check=False)
    seconds = time.monotonic() - start
    if proc.returncode != 0:
        tail = log_file.read_text()[-2000:]
        raise RuntimeError(
            f"AlphaFold 3 exited with {proc.returncode} after {seconds:.0f}s; "
            f"log {log_file}\n{tail}"
        )
    result = parse_af3_output(output_dir, name)
    if result is None:
        raise RuntimeError(f"AlphaFold 3 produced no job directory in {output_dir}")
    if mode == "data_pipeline":
        if result.data_json is None:
            raise RuntimeError(f"AlphaFold 3 data pipeline wrote no *_data.json in "
                               f"{result.job_dir}")
    elif not result.ok:
        raise RuntimeError(f"AlphaFold 3 wrote no model in {result.job_dir}")
    return result

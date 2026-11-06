"""Structured run report (``report.json``, schema ``cofolding-report-v2``)."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

SCHEMA = "cofolding-report-v2"
ENGINES = ("alphafold3", "boltz", "chai1", "protenix", "openfold3",
           "rosettafold3", "intellifold", "opendde", "esmfold2")
STATUSES = ("ok", "failed", "skipped")

# Score keys in order of preference; the first one present is the engine's
# native ranking score.
SCORE_KEYS = (
    "ranking_score",          # AlphaFold 3, Protenix
    "confidence_score",       # Boltz
    "aggregate_score",        # Chai-1
    "sample_ranking_score",   # OpenFold3
)

SAMPLE_FIELDS = (
    "engine", "version", "seed", "rank", "cif", "score", "ptm", "iptm",
    "plddt", "ligand_plddt", "ligand_pae", "chain_ids", "status", "seconds",
)


def _scalar(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 0 or not np.isfinite(arr[0]):
        return None
    return float(arr[0])


def extract_scores(data: Optional[Mapping]) -> Dict[str, Optional[float]]:
    """Pick score/ptm/iptm out of any backend's confidence summary."""
    data = data or {}
    score = None
    for key in SCORE_KEYS:
        if key in data:
            score = _scalar(data[key])
            if score is not None:
                break
    return {
        "score": score,
        "ptm": _scalar(data.get("ptm")),
        "iptm": _scalar(data.get("iptm")),
    }


def to_percent(values: Iterable[float]) -> Optional[float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return None
    mean = float(arr.mean())
    # Boltz/Chai/OF3 write pLDDT in B-factors on a 0-1 scale.
    if float(arr.max()) <= 1.0:
        mean *= 100.0
    return round(mean, 3)


def ligand_pae(
    pae: Optional[Sequence[Sequence[float]]],
    token_chain_ids: Optional[Sequence[str]],
    ligand_chains: Sequence[str],
) -> Optional[float]:
    """Mean ligand<->polymer PAE over both directions (Angstrom)."""
    if pae is None or token_chain_ids is None or not ligand_chains:
        return None
    matrix = np.asarray(pae, dtype=float)
    chains = np.asarray(token_chain_ids)
    if matrix.ndim != 2 or matrix.shape[0] != len(chains):
        return None
    lig = np.isin(chains, list(ligand_chains))
    other = ~lig
    if not lig.any() or not other.any():
        return None
    block = np.concatenate([
        matrix[np.ix_(lig, other)].ravel(),
        matrix[np.ix_(other, lig)].ravel(),
    ])
    return round(float(block.mean()), 3)


@dataclass
class SampleRecord:
    engine: str
    version: Optional[str]
    seed: Optional[int]
    rank: Optional[int]
    cif: str
    score: Optional[float] = None
    ptm: Optional[float] = None
    iptm: Optional[float] = None
    plddt: Optional[float] = None
    ligand_plddt: Optional[float] = None
    ligand_pae: Optional[float] = None
    chain_ids: List[str] = field(default_factory=list)
    status: str = "ok"
    seconds: Optional[float] = None


@dataclass
class EngineRecord:
    engine: str
    version: Optional[str] = None
    status: str = "skipped"
    seconds: Optional[float] = None
    error: Optional[str] = None
    samples: List[SampleRecord] = field(default_factory=list)

    def rank_samples(self) -> None:
        def key(s: SampleRecord):
            return (s.score is None, -(s.score or 0.0))

        for i, sample in enumerate(sorted(self.samples, key=key), start=1):
            sample.rank = i
        self.samples.sort(key=lambda s: s.rank or 0)


def _seed_int(seed: Any) -> Optional[int]:
    text = str(seed)
    digits = "".join(ch for ch in text.split("-")[-1] if ch.isdigit())
    return int(digits) if digits else None


def _load_score_data(score_obj: Any) -> Optional[Mapping]:
    if score_obj is None:
        return None
    data = getattr(score_obj, "data", score_obj)
    if isinstance(data, Mapping):
        return data
    if hasattr(data, "files"):  # np.lib.npyio.NpzFile
        return {k: data[k] for k in data.files}
    return None


def samples_from_output(
    engine: str,
    version: Optional[str],
    output: Mapping,
    output_dir: Path,
    seconds_per_sample: Optional[float] = None,
) -> List[SampleRecord]:
    """Convert an ``abcfold.output.*Output.output`` mapping into samples."""
    samples: List[SampleRecord] = []
    for seed, models in output.items():
        for idx, files in models.items():
            if isinstance(idx, int) and idx < 0:
                continue
            cif = files.get("cif")
            if cif is None:
                continue
            score_obj = None
            for key in ("summary", "json", "scores", "score"):
                if files.get(key) is not None:
                    score_obj = files[key]
                    break
            scores = extract_scores(_load_score_data(score_obj))
            chain_ids = [chain.id for chain in cif.get_chains()]
            ligand_chains = [c.id for c in cif.get_chains() if cif.check_ligand(c)]
            ligand_bfactors = [
                b for values in cif.get_plddt_per_ligand().values() for b in values
            ]
            pae_data = _load_score_data(files.get("af3_pae")) or {}
            cif_path = Path(cif.pathway)
            try:
                cif_rel = cif_path.resolve().relative_to(output_dir.resolve())
                cif_str = cif_rel.as_posix()
            except ValueError:
                cif_str = str(cif_path)
            samples.append(SampleRecord(
                engine=engine,
                version=version,
                seed=_seed_int(seed),
                rank=None,
                cif=cif_str,
                score=scores["score"],
                ptm=scores["ptm"],
                iptm=scores["iptm"],
                plddt=to_percent(cif.plddts),
                ligand_plddt=to_percent(ligand_bfactors),
                ligand_pae=ligand_pae(pae_data.get("pae"),
                                      pae_data.get("token_chain_ids"),
                                      ligand_chains),
                chain_ids=chain_ids,
                seconds=seconds_per_sample,
            ))
    return samples


def build_report(
    name: str,
    engines: Sequence[EngineRecord],
    msa: Optional[Mapping] = None,
    input_json: Optional[str] = None,
    output_dir: Optional[str] = None,
    abcfold_version: Optional[str] = None,
) -> Dict[str, Any]:
    for engine in engines:
        engine.rank_samples()
    requested = [e for e in engines if e.status != "skipped"]
    ok = [e for e in requested if e.status == "ok"]
    if not requested:
        status = "skipped"
    elif len(ok) == len(requested):
        status = "ok"
    elif ok:
        status = "partial"
    else:
        status = "failed"
    samples = [asdict(s) for e in engines for s in e.samples]
    return {
        "schema": SCHEMA,
        "name": name,
        "status": status,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "abcfold_version": abcfold_version,
        "input_json": input_json,
        "output_dir": output_dir,
        "engines": [
            {
                "engine": e.engine,
                "version": e.version,
                "status": e.status,
                "seconds": None if e.seconds is None else round(e.seconds, 2),
                "error": e.error,
                "n_samples": len(e.samples),
                "top_score": e.samples[0].score if e.samples else None,
            }
            for e in engines
        ],
        "samples": samples,
        "msa": dict(msa) if msa else {"source": None, "chains": {}},
    }


def validate_report(report: Mapping) -> List[str]:
    """Return a list of schema violations (empty when valid)."""
    errors: List[str] = []
    if report.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}")
    for key in ("name", "status", "engines", "samples", "msa"):
        if key not in report:
            errors.append(f"missing key {key!r}")
    if report.get("status") not in ("ok", "partial", "failed", "skipped"):
        errors.append(f"bad status {report.get('status')!r}")
    for i, engine in enumerate(report.get("engines", [])):
        if engine.get("engine") not in ENGINES:
            errors.append(f"engines[{i}]: unknown engine {engine.get('engine')!r}")
        if engine.get("status") not in STATUSES:
            errors.append(f"engines[{i}]: bad status {engine.get('status')!r}")
        if engine.get("status") == "failed" and not engine.get("error"):
            errors.append(f"engines[{i}]: failed engine without error")
    for i, sample in enumerate(report.get("samples", [])):
        missing = [f for f in SAMPLE_FIELDS if f not in sample]
        if missing:
            errors.append(f"samples[{i}]: missing {missing}")
        for key in ("score", "ptm", "iptm", "plddt", "ligand_plddt", "ligand_pae",
                    "seconds"):
            value = sample.get(key)
            if value is not None and not isinstance(value, (int, float)):
                errors.append(f"samples[{i}].{key} is not numeric")
    msa = report.get("msa", {})
    if not isinstance(msa, Mapping) or "chains" not in msa:
        errors.append("msa must be an object with 'chains'")
    return errors


def write_report(report: Mapping, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".report.", suffix=".json")
    with os.fdopen(fd, "w") as handle:
        json.dump(report, handle, indent=2)
    os.replace(tmp, path)
    return path

"""Guards against MSAs that are requested but silently lost on the way.

A backend that gets an empty or near-empty MSA still produces a structure,
just a much worse one, so a lost MSA is easy to miss. These checks turn that
into an explicit failure.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

from abcfold.msa.formats import is_gap_row, parse_a3m

MIN_MSA_ROWS = 10
N_MSA_PATTERN = re.compile(r"N_msa\s+(\d+)")


class ShallowMsaError(RuntimeError):
    """An MSA was requested but the backend got fewer rows than allowed."""


def min_msa_rows(config: Optional[Mapping] = None) -> int:
    value = (config or {}).get("min_msa_rows")
    if value in (None, "", "None"):
        return MIN_MSA_ROWS
    return int(str(value))


def msa_rows(unpaired: Optional[str], paired: Optional[str] = None) -> int:
    """Number of distinct non-gap sequences (query included) in both MSAs."""
    seen = set()
    for text in (paired, unpaired):
        for _, seq in parse_a3m(text):
            if not is_gap_row(seq):
                seen.add(seq)
    return len(seen)


def _read(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    return p.read_text() if p.exists() else ""


def entity_msa_text(entity: Mapping) -> tuple[Optional[str], Optional[str]]:
    unpaired = entity.get("unpairedMsa")
    if unpaired is None:
        unpaired = _read(entity.get("unpairedMsaPath"))
    paired = entity.get("pairedMsa")
    if paired is None:
        paired = _read(entity.get("pairedMsaPath"))
    return unpaired, paired


def msa_requested(entity: Mapping) -> bool:
    """True when the input carries a non-empty MSA (or an MSA file path)."""
    unpaired = entity.get("unpairedMsa")
    paired = entity.get("pairedMsa")
    return bool(
        (isinstance(unpaired, str) and unpaired.strip())
        or (isinstance(paired, str) and paired.strip())
        or entity.get("unpairedMsaPath")
        or entity.get("pairedMsaPath")
    )


def shallow_chains(params: Mapping, min_rows: int) -> List[str]:
    """Protein chains whose requested MSA has fewer than ``min_rows`` rows."""
    out = []
    for entry in params.get("sequences", []):
        protein = entry.get("protein")
        if not protein or not msa_requested(protein):
            continue
        rows = msa_rows(*entity_msa_text(protein))
        if rows < min_rows:
            ids = protein.get("id")
            out.append(f"{ids}: {rows} rows")
    return out


def check_input_msa_depth(params: Mapping, min_rows: int,
                          context: str = "input") -> None:
    if min_rows <= 0:
        return
    shallow = shallow_chains(params, min_rows)
    if shallow:
        raise ShallowMsaError(
            f"{context}: MSA requested but too shallow ({', '.join(shallow)}; "
            f"minimum {min_rows}). Check the MSA source, or pass "
            f"--min_msa_rows 0 for genuinely orphan proteins."
        )


def check_protenix_input(protenix_json: Iterable[Mapping], min_rows: int) -> None:
    """Check the MSA files referenced by a Protenix input before running it."""
    if min_rows <= 0:
        return
    problems = []
    for job in protenix_json:
        for entry in job.get("sequences", []):
            chain = entry.get("proteinChain")
            if not chain:
                continue
            for key in ("unpairedMsaPath", "pairedMsaPath"):
                path = chain.get(key)
                if path and not Path(path).exists():
                    problems.append(f"{key} {path} does not exist")
            if chain.get("unpairedMsaPath") or chain.get("unpairedMsa"):
                rows = msa_rows(*entity_msa_text(chain))
                if rows < min_rows:
                    problems.append(f"chain {chain['sequence'][:10]}...: {rows} rows")
    if problems:
        raise ShallowMsaError(
            "Protenix input has a missing or shallow MSA: " + "; ".join(problems)
        )


def parse_n_msa(log_text: str) -> List[int]:
    return [int(m) for m in N_MSA_PATTERN.findall(log_text or "")]


def check_backend_n_msa(engine: str, log_text: str, min_rows: int) -> None:
    """Fail if the backend logged an N_msa below ``min_rows``."""
    if min_rows <= 0:
        return
    values = parse_n_msa(log_text)
    low = [v for v in values if v < min_rows]
    if low:
        raise ShallowMsaError(
            f"{engine} used N_msa={min(low)} although an MSA was provided "
            f"(minimum {min_rows}); the MSA was lost before featurisation."
        )

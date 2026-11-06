"""Minimal client for the ColabFold MMseqs2 API (https://api.colabfold.com).

Follows the protocol of ``colabfold.colabfold.run_mmseqs2``: submit a ticket,
poll it, download a tarball of A3M files whose records are grouped by query
id (101, 102, ...).
"""

from __future__ import annotations

import io
import logging
import random
import tarfile
import time
from typing import Callable, Dict, List, Optional, Sequence

import requests

from abcfold.msa.formats import format_a3m, parse_a3m

logger = logging.getLogger("logger")

DEFAULT_URL = "https://api.colabfold.com"
UNPAIRED_FILES = ("uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m")
PAIRED_FILES = ("pair.a3m",)
FIRST_ID = 101


class ColabFoldError(RuntimeError):
    pass


def split_colabfold_a3m(text: str) -> Dict[int, str]:
    """Split a ColabFold result A3M into per-query blocks keyed by query id."""
    blocks: Dict[int, List[str]] = {}
    current: Optional[int] = None
    expect_id = True
    for line in text.splitlines(keepends=True):
        if "\x00" in line:
            line = line.replace("\x00", "")
            expect_id = True
        if not line:
            continue
        if line.startswith(">") and expect_id:
            try:
                current = int(line[1:].strip())
            except ValueError:
                pass
            expect_id = False
            blocks.setdefault(current, [])  # type: ignore[arg-type]
        if current is not None:
            blocks[current].append(line)
    return {k: "".join(v) for k, v in blocks.items()}


def merge_query_blocks(texts: Sequence[str], query_id: int) -> str:
    """Concatenate per-database blocks of one query, keeping a single query row."""
    records = []
    for i, text in enumerate(texts):
        for j, (header, seq) in enumerate(parse_a3m(text)):
            if j == 0 and header.strip() == str(query_id):
                if i > 0:
                    continue
                header = "query"
            records.append((header, seq))
    return format_a3m(records)


class ColabFoldClient:
    def __init__(
        self,
        url: str = DEFAULT_URL,
        user_agent: str = "abcfold",
        session=None,
        poll_seconds: float = 5.0,
        timeout: float = 3600.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.url = url.rstrip("/")
        self.session = session or requests.Session()
        self.headers = {"User-Agent": user_agent}
        self.poll_seconds = poll_seconds
        self.timeout = timeout
        self.sleep = sleep

    def _json(self, response) -> dict:
        try:
            return response.json()
        except ValueError:
            return {"status": "ERROR", "error": response.text[:500]}

    def _submit(self, endpoint: str, sequences: Sequence[str], mode: str) -> dict:
        query = "".join(
            f">{FIRST_ID + i}\n{seq}\n" for i, seq in enumerate(sequences)
        )
        while True:
            out = self._json(
                self.session.post(
                    f"{self.url}/{endpoint}",
                    data={"q": query, "mode": mode},
                    headers=self.headers,
                    timeout=60,
                )
            )
            if out.get("status") not in ("UNKNOWN", "RATELIMIT"):
                return out
            self.sleep(self.poll_seconds + random.random() * self.poll_seconds)

    def _wait(self, ticket: dict) -> str:
        start = time.monotonic()
        status = ticket.get("status")
        ticket_id = ticket.get("id")
        while status in ("UNKNOWN", "RUNNING", "PENDING"):
            if time.monotonic() - start > self.timeout:
                raise ColabFoldError(f"MSA ticket {ticket_id} timed out")
            self.sleep(self.poll_seconds + random.random() * self.poll_seconds)
            status = self._json(
                self.session.get(
                    f"{self.url}/ticket/{ticket_id}", headers=self.headers,
                    timeout=60,
                )
            ).get("status")
        if status != "COMPLETE":
            raise ColabFoldError(
                f"ColabFold MSA server returned status {status}: {ticket}"
            )
        if ticket_id is None:
            raise ColabFoldError(f"ColabFold MSA ticket carries no id: {ticket}")
        return str(ticket_id)

    def _download(self, ticket_id: str) -> Dict[str, str]:
        response = self.session.get(
            f"{self.url}/result/download/{ticket_id}", headers=self.headers,
            timeout=600,
        )
        files: Dict[str, str] = {}
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    handle = tar.extractfile(member)
                    if handle is not None:
                        files[member.name.split("/")[-1]] = handle.read().decode()
        return files

    def _run(self, endpoint: str, sequences: Sequence[str], mode: str,
             names: Sequence[str]) -> List[str]:
        unique = list(dict.fromkeys(sequences))
        ticket = self._submit(endpoint, unique, mode)
        if ticket.get("status") in ("ERROR", "MAINTENANCE"):
            raise ColabFoldError(f"ColabFold MSA server error: {ticket}")
        files = self._download(self._wait(ticket))
        per_file = [split_colabfold_a3m(files[n]) for n in names if n in files]
        if not per_file:
            raise ColabFoldError(f"No A3M files in ColabFold result: {list(files)}")
        out = []
        for seq in sequences:
            qid = FIRST_ID + unique.index(seq)
            out.append(merge_query_blocks([blocks.get(qid, "") for blocks in per_file],
                                          qid))
        return out

    def unpaired(self, sequences: Sequence[str], use_env: bool = True) -> List[str]:
        """One unpaired A3M per input sequence (UniRef + environmental DBs)."""
        mode = "env" if use_env else "all"
        names = UNPAIRED_FILES if use_env else UNPAIRED_FILES[:1]
        return self._run("ticket/msa", sequences, mode, names)

    def paired(self, sequences: Sequence[str], strategy: str = "greedy") -> List[str]:
        """Row-aligned paired A3Ms, one per chain, in input order."""
        mode = "pairgreedy" if strategy == "greedy" else "paircomplete"
        # As in ColabFold, pairing runs over the unique sequences; copies of a
        # homomeric chain share the same paired rows.
        return self._run("ticket/pair", sequences, mode, PAIRED_FILES)

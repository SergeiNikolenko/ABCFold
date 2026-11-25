# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

# Notice: This file has been modified to include a wrapper around the
# `run_inference` function to allow for the PAE (Predicted Aligned Error)
# to be output and to serve local/custom templates from a template CIF store.

"""Command line interface."""

import contextlib
import inspect
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import typer
from chai_lab.chai1 import run_inference

logging.basicConfig(level=logging.INFO)

_ORIG_DOWNLOAD_CIF_FILE: Callable[..., Path] | None = None
_ACTIVE_TEMPLATE_STORE: Path | None = None
_ORIG_LOAD_EXPORTED: Callable[..., Any] | None = None
_ORIG_MAKE_FEATURE_CONTEXT: Callable[..., Any] | None = None
_ORIG_GET_ESM_FOR_SEQS: Callable[..., Any] | None = None
_EXPORTED_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_LAST_FEATURE_CONTEXT: tuple[tuple[Any, ...], Any] | None = None
_ESM_SEQ_CACHE: dict[str, Any] = {}
_ESM_SEQ_CACHE_MAX = 64

CITATION = """
@article{Chai-1-Technical-Report,
    title        = {Chai-1: Decoding the molecular interactions of life},
    author       = {{Chai Discovery}},
    year         = 2024,
    journal      = {bioRxiv},
    publisher    = {Cold Spring Harbor Laboratory},
    doi          = {10.1101/2024.10.10.615955},
    url          = {https://www.biorxiv.org/content/early/2024/10/11/2024.10.10.615955},
    elocation-id = {2024.10.10.615955},
    eprint       = {https://www.biorxiv.org/content/early/2024/10/11/2024.10.1\
0.615955.full.pdf}
}
""".strip()


def citation():
    """Print citation information."""
    typer.echo(CITATION)


def _read_fasta_proteins(fasta_file: Path) -> dict[str, str]:
    """Return {chain_id: sequence} for the ``>protein|<id>`` records."""
    seqs: dict[str, str] = {}
    chain: str | None = None
    for line in Path(fasta_file).read_text().splitlines():
        if line.startswith(">"):
            parts = line[1:].split("|")
            chain = parts[1] if len(parts) > 1 and parts[0] == "protein" else None
        elif chain is not None and line.strip():
            seqs[chain] = seqs.get(chain, "") + line.strip()
    return seqs


def _subject_span(query_seq: str, template_seq: str, kalign_fn: Callable):
    """Template [start, end) that aligns to the query, via chai-lab kalign."""
    alignment = kalign_fn(ref=query_seq, query=template_seq)
    if alignment is None:
        return None
    t_idx = -1
    first = last = None
    for q_ch, t_ch in zip(alignment.reference_aligned, alignment.query_aligned):
        if t_ch != "-":
            t_idx += 1
            if q_ch != "-":
                if first is None:
                    first = t_idx
                last = t_idx
    if first is None or last is None:
        return None
    return first, last + 1


def _build_custom_m8(
    store: Path,
    fasta_file: Path,
    existing_m8: Path | None,
    out_m8: Path,
) -> Path | None:
    """Assemble the Chai m8, adding kalign-derived rows for custom templates."""
    import gemmi

    rows: list[str] = []
    manifest_path = store / "custom_templates.json"
    if manifest_path.is_file():
        from chai_lab.tools.kalign import kalign_query_to_reference

        manifest = json.loads(manifest_path.read_text())
        queries = _read_fasta_proteins(fasta_file)
        for synth_id, info in manifest.items():
            cif_gz = store / f"{synth_id}.cif.gz"
            if not cif_gz.is_file():
                continue
            try:
                structure = gemmi.read_structure(str(cif_gz))
                tchain = info["template_chain"]
                tseq = (
                    structure[0][tchain]
                    .get_polymer()
                    .make_one_letter_sequence()
                    .replace("-", "")
                )
            except Exception as exc:  # noqa: BLE001
                logging.warning("Custom template %s unreadable: %s", synth_id, exc)
                continue
            for qchain in info.get("query_chains", []):
                qseq = queries.get(qchain)
                if not qseq:
                    continue
                span = _subject_span(qseq, tseq, kalign_query_to_reference)
                if span is None:
                    logging.warning(
                        "Custom template %s did not align to chain %s; skipping",
                        synth_id,
                        qchain,
                    )
                    continue
                s_start, s_end = span
                rows.append(
                    "\t".join(
                        [
                            str(qchain),
                            f"{synth_id}_{tchain}",
                            "100.0",
                            str(s_end - s_start),
                            "0",
                            "0",
                            "1",
                            str(len(qseq)),
                            str(s_start + 1),
                            str(s_end),
                            "1e-9",
                            "100.0",
                            "custom",
                        ]
                    )
                )
        if rows:
            logging.info("Added %d custom template hit(s) to the m8", len(rows))

    lines: list[str] = []
    if existing_m8 is not None and Path(existing_m8).is_file():
        lines += [
            ln for ln in Path(existing_m8).read_text().splitlines() if ln.strip()
        ]
    lines += rows
    if not lines:
        return None
    out_m8.write_text("\n".join(lines) + "\n")
    return out_m8


def _install_template_store(store: Path | None) -> None:
    """Point the rcsb.download_cif_file hook at ``store`` (idempotent)."""
    global _ORIG_DOWNLOAD_CIF_FILE, _ACTIVE_TEMPLATE_STORE
    import chai_lab.data.io.rcsb as rcsb

    if _ORIG_DOWNLOAD_CIF_FILE is None:
        _ORIG_DOWNLOAD_CIF_FILE = rcsb.download_cif_file

        def _patched(pdb_id: str, directory: Path) -> Path:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            outfile = directory / f"{pdb_id}.cif.gz"
            active = _ACTIVE_TEMPLATE_STORE
            if active is not None and not outfile.exists():
                local = active / f"{pdb_id}.cif.gz"
                if local.exists():
                    shutil.copyfile(local, outfile)
                    logging.info("Using local template CIF for %s", pdb_id)
                    return outfile
            assert _ORIG_DOWNLOAD_CIF_FILE is not None
            return _ORIG_DOWNLOAD_CIF_FILE(pdb_id, directory)

        rcsb.download_cif_file = _patched

    _ACTIVE_TEMPLATE_STORE = store


@contextlib.contextmanager
def _template_store_context(store: Path | None) -> Iterator[None]:
    prev = _ACTIVE_TEMPLATE_STORE
    if store is not None:
        _install_template_store(store)
    try:
        yield
    finally:
        if store is not None:
            _install_template_store(prev)


def normalize_device(device: str | None) -> str | None:
    if device is None:
        return None
    if device == "cpu":
        return "cpu"
    if device == "all":
        return "cuda"
    if "," in device:
        device = device.split(",")[0]
    if device.isdigit():
        return f"cuda:{device}"
    return device


HIGH_VRAM_THRESHOLD_BYTES = 24 * (1024**3)


def resolve_low_memory(
    low_memory: bool | None,
    device: str | None = None,
) -> bool:
    """Choose Chai-1 low_memory automatically from GPU VRAM when unset.

    low_memory=True offloads trunk/embedding tensors back to CPU across PCIe
    every step. GPUs with >= 24 GiB VRAM (A100, H100, L40S, 3090/4090) keep
    them resident on GPU cleanly and run substantially faster.
    """
    if low_memory is not None:
        return bool(low_memory)
    norm = normalize_device(device)
    if norm == "cpu":
        return True
    try:
        import torch

        if not torch.cuda.is_available():
            return True
        idx = 0
        if norm and norm.startswith("cuda:"):
            idx = int(norm.split(":", 1)[1])
        total_mem = torch.cuda.get_device_properties(idx).total_memory
        return total_mem < HIGH_VRAM_THRESHOLD_BYTES
    except Exception:  # noqa: BLE001
        return True


def _enable_fast_cuda_math() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception:  # noqa: BLE001
        pass


def _install_inference_caches() -> None:
    """Cache Chai-1 exported TorchScript models and per-complex feature context.

    Chai-1 loads all 6 TorchScript checkpoints from disk on every run_inference
    call and recomputes ESM-2 embeddings + MSA parsing on every seed. Caching
    load_exported across tasks (zero extra peak VRAM, because Chai-1 already
    holds all 6 modules concurrently during folding) and caching the most recent
    feature_context across seeds of the same complex eliminates repeated disk
    I/O, MSA parsing, and ESM-2 inference.
    """
    global _ORIG_LOAD_EXPORTED, _ORIG_MAKE_FEATURE_CONTEXT
    import chai_lab.chai1 as chai1_mod

    if _ORIG_LOAD_EXPORTED is None and hasattr(chai1_mod, "load_exported"):
        _ORIG_LOAD_EXPORTED = chai1_mod.load_exported

        def _cached_load_exported(comp_key: str, device: Any) -> Any:
            key = (str(comp_key), str(device))
            cached = _EXPORTED_MODEL_CACHE.get(key)
            if cached is not None:
                return cached
            assert _ORIG_LOAD_EXPORTED is not None
            module = _ORIG_LOAD_EXPORTED(comp_key, device)
            _EXPORTED_MODEL_CACHE[key] = module
            return module

        chai1_mod.load_exported = _cached_load_exported

    if _ORIG_MAKE_FEATURE_CONTEXT is None and hasattr(
        chai1_mod, "make_all_atom_feature_context"
    ):
        _ORIG_MAKE_FEATURE_CONTEXT = chai1_mod.make_all_atom_feature_context

        def _cached_make_context(*args: Any, **kwargs: Any) -> Any:
            global _LAST_FEATURE_CONTEXT
            assert _ORIG_MAKE_FEATURE_CONTEXT is not None
            use_msa_server = kwargs.get("use_msa_server", False)
            use_templates_server = kwargs.get("use_templates_server", False)
            if use_msa_server or use_templates_server:
                return _ORIG_MAKE_FEATURE_CONTEXT(*args, **kwargs)
            fasta_arg = args[0] if args else kwargs.get("fasta_file")
            msa_arg = kwargs.get("msa_directory")
            constr_arg = kwargs.get("constraint_path")
            tmpl_arg = kwargs.get("templates_path")
            key = (
                str(Path(fasta_arg).resolve()) if fasta_arg else None,
                str(Path(msa_arg).resolve()) if msa_arg else None,
                str(Path(constr_arg).resolve()) if constr_arg else None,
                str(Path(tmpl_arg).resolve()) if tmpl_arg else None,
                bool(kwargs.get("use_esm_embeddings", True)),
                str(kwargs.get("esm_device", "")),
            )
            if _LAST_FEATURE_CONTEXT is not None and _LAST_FEATURE_CONTEXT[0] == key:
                logging.info("Reusing cached Chai-1 feature context (ESM + MSA)")
                return _LAST_FEATURE_CONTEXT[1]
            ctx = _ORIG_MAKE_FEATURE_CONTEXT(*args, **kwargs)
            _LAST_FEATURE_CONTEXT = (key, ctx)
            return ctx

        chai1_mod.make_all_atom_feature_context = _cached_make_context

    global _ORIG_GET_ESM_FOR_SEQS
    if _ORIG_GET_ESM_FOR_SEQS is None:
        try:
            import chai_lab.data.dataset.embeddings.esm as esm_mod
        except Exception:  # noqa: BLE001
            esm_mod = None
        if esm_mod is not None and hasattr(
            esm_mod, "_get_esm_contexts_for_sequences"
        ):
            _ORIG_GET_ESM_FOR_SEQS = esm_mod._get_esm_contexts_for_sequences

            def _cached_get_esm(
                prot_sequences: set[str], device: Any
            ) -> dict[str, Any]:
                assert _ORIG_GET_ESM_FOR_SEQS is not None
                missing = {seq for seq in prot_sequences if seq not in _ESM_SEQ_CACHE}
                if missing:
                    computed = _ORIG_GET_ESM_FOR_SEQS(
                        prot_sequences=missing, device=device
                    )
                    for seq, emb in computed.items():
                        if len(_ESM_SEQ_CACHE) >= _ESM_SEQ_CACHE_MAX:
                            _ESM_SEQ_CACHE.pop(next(iter(_ESM_SEQ_CACHE)))
                        _ESM_SEQ_CACHE[seq] = emb
                elif prot_sequences:
                    logging.info(
                        "Reusing cached ESM-2 embeddings for %d sequence(s)",
                        len(prot_sequences),
                    )
                return {seq: _ESM_SEQ_CACHE[seq] for seq in prot_sequences}

            esm_mod._get_esm_contexts_for_sequences = _cached_get_esm


def build_run_inference_kwargs(
    fasta_file: Path,
    *,
    output_dir: Path,
    use_esm_embeddings: bool = True,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_directory: Path | None = None,
    constraint_path: Path | None = None,
    use_templates_server: bool = False,
    template_hits_path: Path | None = None,
    recycle_msa_subsample: int = 0,
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    num_diffn_samples: int = 5,
    num_trunk_samples: int = 1,
    seed: int | None = None,
    device: str | None = None,
    low_memory: bool | None = None,
) -> dict[str, Any]:
    norm_device = normalize_device(device)
    kwargs = {
        "fasta_file": fasta_file,
        "output_dir": output_dir,
        "use_esm_embeddings": use_esm_embeddings,
        "use_msa_server": use_msa_server,
        "msa_server_url": msa_server_url,
        "msa_directory": msa_directory,
        "constraint_path": constraint_path,
        "use_templates_server": use_templates_server,
        "template_hits_path": template_hits_path,
        "recycle_msa_subsample": recycle_msa_subsample,
        "num_trunk_recycles": num_trunk_recycles,
        "num_diffn_timesteps": num_diffn_timesteps,
        "num_diffn_samples": num_diffn_samples,
        "num_trunk_samples": num_trunk_samples,
        "seed": seed,
        "device": norm_device,
        "low_memory": resolve_low_memory(low_memory, norm_device),
    }
    supported = inspect.signature(run_inference).parameters
    return {key: value for key, value in kwargs.items() if key in supported}


def run_inference_wrapper(
    fasta_file: Path,
    *,
    output_dir: Path,
    use_esm_embeddings: bool = True,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_directory: Path | None = None,
    constraint_path: Path | None = None,
    use_templates_server: bool = False,
    template_hits_path: Path | None = None,
    template_cif_store: Path | None = None,
    recycle_msa_subsample: int = 0,
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    num_diffn_samples: int = 5,
    num_trunk_samples: int = 1,
    seed: int | None = None,
    device: str | None = None,
    low_memory: bool | None = None,
):
    _enable_fast_cuda_math()
    _install_inference_caches()
    effective_m8 = template_hits_path
    if template_cif_store is not None and Path(template_cif_store).is_dir():
        store = Path(template_cif_store)
        if (store / "custom_templates.json").is_file():
            out_m8 = Path(output_dir) / "combined_templates.m8"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            effective_m8 = _build_custom_m8(
                store, Path(fasta_file), template_hits_path, out_m8
            )
    store_path = (
        Path(template_cif_store)
        if template_cif_store is not None and Path(template_cif_store).is_dir()
        else None
    )
    with _template_store_context(store_path):
        result = run_inference(
            **build_run_inference_kwargs(
                fasta_file,
                output_dir=output_dir,
                use_esm_embeddings=use_esm_embeddings,
                use_msa_server=use_msa_server,
                msa_server_url=msa_server_url,
                msa_directory=msa_directory,
                constraint_path=constraint_path,
                use_templates_server=use_templates_server,
                template_hits_path=effective_m8,
                recycle_msa_subsample=recycle_msa_subsample,
                num_trunk_recycles=num_trunk_recycles,
                num_diffn_timesteps=num_diffn_timesteps,
                num_diffn_samples=num_diffn_samples,
                num_trunk_samples=num_trunk_samples,
                seed=seed,
                device=device,
                low_memory=low_memory,
            )
        )

    np.save(f"{output_dir}/pae_scores.npy", result.pae)
    return result


def cli():
    app = typer.Typer()
    app.command("fold", help="Run Chai-1 to fold a complex.")(run_inference_wrapper)
    app.command("citation", help="Print citation information")(citation)
    app()


if __name__ == "__main__":
    cli()

"""Persistent Boltz-2 GPU worker executed inside the isolated Boltz environment.

Loads Boltz2 weights once onto GPU and runs process_inputs once across all
unique YAML inputs, then iterates over every requested seed without repeating
Python/CUDA startup, MSA parsing, CCD loading, or checkpoint reads.

Usage:
    python boltz_worker.py <manifest.json>
"""

from __future__ import annotations

import contextlib
import gc
import importlib.util
import inspect
import json
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator


@contextlib.contextmanager
def _skip_param_init(torch_mod: Any) -> Iterator[None]:
    """Temporarily make torch.nn.init.* no-ops during checkpoint load."""
    nn_init = torch_mod.nn.init
    saved: dict[str, Any] = {}
    noop = lambda tensor, *args, **kwargs: tensor  # noqa: E731
    for name in (
        "kaiming_uniform_",
        "kaiming_normal_",
        "xavier_uniform_",
        "xavier_normal_",
        "normal_",
        "uniform_",
        "orthogonal_",
        "trunc_normal_",
        "zeros_",
        "ones_",
        "constant_",
    ):
        fn = getattr(nn_init, name, None)
        if callable(fn):
            saved[name] = fn
            setattr(nn_init, name, noop)
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(nn_init, name, fn)


def _fast_load_boltz_checkpoint(
    torch_mod: Any,
    model_cls: Any,
    ckpt_path: Path,
    *,
    device: str,
    strict: bool = True,
    **kwargs: Any,
) -> Any:
    """Load a Boltz2 checkpoint while skipping redundant random weight init."""
    with _skip_param_init(torch_mod):
        return model_cls.load_from_checkpoint(
            ckpt_path,
            strict=strict,
            map_location=device,
            **kwargs,
        )


def _install_msa_parse_cache() -> None:
    """Cache parsed A3M/CSV MSAs in memory across targets in the same batch.

    In many-ligands screening, every ligand YAML references the exact same
    receptor MSA file; caching avoids re-parsing a 16k-row A3M N times.
    """
    cache: dict[tuple[str, str, int | None], Any] = {}
    for mod_name, fn_name in (
        ("boltz.data.parse.a3m", "parse_a3m"),
        ("boltz.data.parse.csv", "parse_csv"),
    ):
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            orig = getattr(mod, fn_name, None)
            if orig is None or getattr(orig, "_abcfold_cached", False):
                continue

            def _make_cached(f: Any, tag: str) -> Any:
                def _wrapped(path: Any, *args: Any, **kwargs: Any) -> Any:
                    max_seqs = kwargs.get("max_seqs")
                    if max_seqs is None and len(args) >= 2:
                        max_seqs = args[1]
                    key = (tag, str(Path(path).resolve()), max_seqs)
                    if key in cache:
                        return cache[key]
                    val = f(path, *args, **kwargs)
                    if len(cache) < 64:
                        cache[key] = val
                    return val

                _wrapped._abcfold_cached = True  # type: ignore[attr-defined]
                return _wrapped

            setattr(mod, fn_name, _make_cached(orig, mod_name))
        except Exception:  # noqa: BLE001
            continue


def _supports_kernels(torch_mod: Any) -> bool:
    if not torch_mod.cuda.is_available():
        return False
    try:
        major = torch_mod.cuda.get_device_properties(0).major
        if major < 8:
            return False
        return (
            importlib.util.find_spec("cuequivariance_ops_torch") is not None
            or importlib.util.find_spec("cuequivariance_ops_cu12") is not None
        )
    except Exception:  # noqa: BLE001
        return False


def _call_matching(fn: Any, **kwargs: Any) -> Any:
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)
    allowed = set(sig.parameters)
    return fn(**{k: v for k, v in kwargs.items() if k in allowed})


def main() -> int:
    manifest_path = Path(sys.argv[1])
    spec = json.loads(manifest_path.read_text())

    import torch
    from boltz.data import const
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.types import Manifest
    from boltz.data.write.writer import BoltzAffinityWriter, BoltzWriter
    from boltz.main import (
        Boltz2DiffusionParams,
        BoltzProcessedInput,
        BoltzSteeringParams,
        MSAModuleArgs,
        PairformerArgsV2,
        check_inputs,
        download_boltz2,
        process_inputs,
    )
    from boltz.model.models.boltz2 import Boltz2
    from pytorch_lightning import Trainer, seed_everything

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    _install_msa_parse_cache()

    cache = Path(spec["cache"]).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    if not (cache / "boltz2_conf.ckpt").exists() or not (cache / "mols").exists():
        download_boltz2(cache)

    shared_work_dir = Path(spec["shared_work_dir"])
    shared_work_dir.mkdir(parents=True, exist_ok=True)

    inputs_source = Path(spec["inputs_source"])
    if inputs_source.is_dir():
        data_paths = sorted(
            p
            for p in inputs_source.iterdir()
            if p.is_file() and p.suffix.lower() in (".yaml", ".yml", ".fasta", ".fa")
        )
        if not data_paths:
            raise RuntimeError(f"No .yaml/.fasta inputs found in {inputs_source}")
    else:
        data_paths = check_inputs(inputs_source)

    method = spec.get("method")
    if method is not None and method.lower() not in const.method_types_ids:
        raise ValueError(
            f"Unsupported Boltz method {method!r}; "
            f"supported: {list(const.method_types_ids.keys())}"
        )

    preprocessing_threads = max(1, int(spec.get("preprocessing_threads") or 4))
    num_workers = max(0, int(spec.get("num_workers") or 2))
    diffusion_samples = max(1, int(spec.get("diffusion_samples") or 5))
    recycling_steps = max(1, int(spec.get("recycling_steps") or 10))
    sampling_steps = max(1, int(spec.get("sampling_steps") or 200))
    max_parallel_samples = spec.get("max_parallel_samples")
    if max_parallel_samples is None:
        max_parallel_samples = min(diffusion_samples, 5)
    else:
        max_parallel_samples = max(1, int(max_parallel_samples))
    use_potentials = bool(spec.get("use_potentials", False))
    use_kernels_req = spec.get("use_kernels")
    use_kernels = (
        _supports_kernels(torch)
        if use_kernels_req is None
        else bool(use_kernels_req)
    )

    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"

    # Preserve RDKit atom properties (e.g. "name") if Boltz multiprocessing
    # pickles Mol objects across preprocessing workers.
    try:
        from rdkit import Chem

        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    except Exception:  # noqa: BLE001
        pass

    # Parse MSA, templates, constraints, and ligands ONCE for all seeds/targets.
    # Use 1 preprocessing thread when multiprocessing would otherwise lose RDKit
    # properties in spawned workers on newer RDKit builds; fall back cleanly if
    # parallel preprocessing yields empty records.
    eff_pre_threads = preprocessing_threads if len(data_paths) <= 1 else 1
    _call_matching(
        process_inputs,
        data=data_paths,
        out_dir=shared_work_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=False,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        boltz2=True,
        preprocessing_threads=eff_pre_threads,
        max_msa_seqs=8192,
    )

    processed_dir = shared_work_dir / "processed"
    manifest = Manifest.load(processed_dir / "manifest.json")
    if not manifest.records:
        raise RuntimeError(
            f"Boltz process_inputs produced 0 valid records from "
            f"{len(data_paths)} input(s)"
        )
    processed = BoltzProcessedInput(
        manifest=manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(
            processed_dir / "constraints"
            if (processed_dir / "constraints").exists()
            else None
        ),
        template_dir=(
            processed_dir / "templates"
            if (processed_dir / "templates").exists()
            else None
        ),
        extra_mols_dir=(
            processed_dir / "mols" if (processed_dir / "mols").exists() else None
        ),
    )

    diffusion_params = Boltz2DiffusionParams()
    diffusion_params.step_scale = float(spec.get("step_scale") or 1.5)
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs(
        subsample_msa=False,
        num_subsampled_msa=1024,
        use_paired_feature=True,
    )
    steering_args = BoltzSteeringParams()
    steering_args.fk_steering = use_potentials
    steering_args.physical_guidance_update = use_potentials

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "max_parallel_samples": max_parallel_samples,
        "write_confidence_summary": True,
        "write_full_pae": True,
        "write_full_pde": True,
    }

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    precision: Any = "bf16-mixed" if accelerator == "gpu" else 32

    # Load Boltz2 structure checkpoint ONCE and reuse across every seed.
    target_device = "cuda:0" if accelerator == "gpu" else "cpu"
    model_module = _fast_load_boltz_checkpoint(
        torch,
        Boltz2,
        cache / "boltz2_conf.ckpt",
        device=target_device,
        strict=True,
        predict_args=predict_args,
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=use_kernels,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )
    model_module.eval()

    class _AsyncBoltzWriter(BoltzWriter):
        """Overlap mmCIF/NPZ/JSON disk writes with GPU compute of next batch."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._io_pool = ThreadPoolExecutor(max_workers=1)
            self._io_futs: list[Future[Any]] = []

        def write_on_batch_end(
            self,
            trainer: Any,
            pl_module: Any,
            prediction: Any,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            # Detach/move tensors to CPU before handing off to writer thread.
            def _to_cpu(obj: Any) -> Any:
                if isinstance(obj, torch.Tensor):
                    return obj.detach().cpu()
                if isinstance(obj, dict):
                    return {k: _to_cpu(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_to_cpu(v) for v in obj]
                if isinstance(obj, tuple):
                    return tuple(_to_cpu(v) for v in obj)
                return obj

            cpu_pred = _to_cpu(prediction)
            fut = self._io_pool.submit(
                super().write_on_batch_end,
                trainer,
                pl_module,
                cpu_pred,
                *args,
                **kwargs,
            )
            self._io_futs.append(fut)

        def flush(self) -> None:
            try:
                for fut in self._io_futs:
                    fut.result()
            finally:
                self._io_futs.clear()
                self._io_pool.shutdown(wait=True)

    seed_record_ids_raw = spec.get("seed_record_ids")
    seed_record_ids: dict[int, set[str]] | None = (
        {int(k): set(v) for k, v in seed_record_ids_raw.items()}
        if seed_record_ids_raw
        else None
    )

    def _manifest_for_seed(seed_val: int) -> Any:
        if seed_record_ids is None or seed_val not in seed_record_ids:
            return processed.manifest
        allowed = seed_record_ids[seed_val]
        filtered = [r for r in processed.manifest.records if r.id in allowed]
        return Manifest(filtered)

    def _eff_workers(n_records: int) -> int:
        # Forking 2 multi-GB PyTorch worker processes for 1-2 records costs
        # several seconds per seed while in-process loading takes ~50 ms.
        return 0 if n_records <= 2 else num_workers

    shared_data_module = (
        None
        if seed_record_ids is not None
        else Boltz2InferenceDataModule(
            manifest=processed.manifest,
            target_dir=processed.targets_dir,
            msa_dir=processed.msa_dir,
            mol_dir=mol_dir,
            num_workers=_eff_workers(len(processed.manifest.records)),
            constraints_dir=processed.constraints_dir,
            template_dir=processed.template_dir,
            extra_mols_dir=processed.extra_mols_dir,
            override_method=method,
        )
    )

    seeds = [int(s) for s in spec["seeds"]]
    seed_out_dirs = {int(k): Path(v) for k, v in spec["seed_out_dirs"].items()}

    for seed in seeds:
        seed_manifest = _manifest_for_seed(seed)
        if not seed_manifest.records:
            continue
        seed_everything(seed)
        seed_out = seed_out_dirs[seed]
        predictions_dir = seed_out / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        dm = (
            shared_data_module
            if shared_data_module is not None
            else Boltz2InferenceDataModule(
                manifest=seed_manifest,
                target_dir=processed.targets_dir,
                msa_dir=processed.msa_dir,
                mol_dir=mol_dir,
                num_workers=_eff_workers(len(seed_manifest.records)),
                constraints_dir=processed.constraints_dir,
                template_dir=processed.template_dir,
                extra_mols_dir=processed.extra_mols_dir,
                override_method=method,
            )
        )
        pred_writer = _AsyncBoltzWriter(
            data_dir=processed.targets_dir,
            output_dir=predictions_dir,
            output_format="mmcif",
            boltz2=True,
            write_embeddings=False,
        )
        trainer = Trainer(
            default_root_dir=seed_out,
            strategy="auto",
            callbacks=[pred_writer],
            accelerator=accelerator,
            devices=1,
            precision=precision,
            logger=False,
            enable_progress_bar=False,
        )
        try:
            trainer.predict(
                model_module,
                datamodule=dm,
                return_predictions=False,
            )
        finally:
            pred_writer.flush()

    # If any YAML requested native Boltz-2 affinity, load boltz2_aff.ckpt ONCE.
    if any(r.affinity for r in manifest.records):
        del model_module
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        aff_ckpt = cache / "boltz2_aff.ckpt"
        if aff_ckpt.exists():
            predict_aff_args = {
                "recycling_steps": 5,
                "sampling_steps": 200,
                "diffusion_samples": 5,
                "max_parallel_samples": 1,
                "write_confidence_summary": False,
                "write_full_pae": False,
                "write_full_pde": False,
            }
            aff_steering = BoltzSteeringParams()
            aff_steering.fk_steering = False
            aff_steering.physical_guidance_update = False
            aff_steering.contact_guidance_update = False
            aff_model = _fast_load_boltz_checkpoint(
                torch,
                Boltz2,
                aff_ckpt,
                device=target_device,
                strict=True,
                predict_args=predict_aff_args,
                diffusion_process_args=asdict(diffusion_params),
                ema=False,
                use_kernels=use_kernels,
                pairformer_args=asdict(pairformer_args),
                msa_args=asdict(msa_args),
                steering_args=asdict(aff_steering),
            )
            aff_model.eval()
            for seed in seeds:
                seed_manifest = _manifest_for_seed(seed)
                aff_records = [r for r in seed_manifest.records if r.affinity]
                if not aff_records:
                    continue
                seed_everything(seed)
                seed_out = seed_out_dirs[seed]
                predictions_dir = seed_out / "predictions"
                aff_dm = Boltz2InferenceDataModule(
                    manifest=Manifest(aff_records),
                    target_dir=predictions_dir,
                    msa_dir=processed.msa_dir,
                    mol_dir=mol_dir,
                    num_workers=_eff_workers(len(aff_records)),
                    constraints_dir=processed.constraints_dir,
                    template_dir=processed.template_dir,
                    extra_mols_dir=processed.extra_mols_dir,
                    override_method="other",
                    affinity=True,
                )
                aff_writer = BoltzAffinityWriter(
                    data_dir=processed.targets_dir,
                    output_dir=predictions_dir,
                )
                trainer = Trainer(
                    default_root_dir=seed_out,
                    strategy="auto",
                    callbacks=[aff_writer],
                    accelerator=accelerator,
                    devices=1,
                    precision=precision,
                    logger=False,
                    enable_progress_bar=False,
                )
                trainer.predict(
                    aff_model,
                    datamodule=aff_dm,
                    return_predictions=False,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

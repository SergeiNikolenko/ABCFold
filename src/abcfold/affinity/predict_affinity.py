# Adapted from Boltzina (https://github.com/ohuelab/boltzina), MIT License,
# and extended to run directly on stock boltz==2.2.1 without requiring the
# external boltzina package (whose PyPI pins conflict with boltz[cuda]==2.2.1).

import gc
import importlib.util
import inspect
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Union

import torch
from boltz.data.types import Manifest
from boltz.main import (Boltz2DiffusionParams, MSAModuleArgs, PairformerArgsV2,
                        get_cache_path)
from pytorch_lightning import Trainer, seed_everything

try:
    from boltzina.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltzina.data.write.writer import BoltzAffinityWriter
    from boltzina.model.models.boltz2 import Boltz2 as _BaseBoltz2

    _HAS_BOLTZINA = True
except ImportError:
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.write.writer import BoltzAffinityWriter
    from boltz.model.models.boltz2 import Boltz2 as _BaseBoltz2

    _HAS_BOLTZINA = False


def _score_affinity_on_fixed_coords(
    model: Any,
    batch: dict[str, Any],
    out: dict[str, Any],
) -> dict[str, Any]:
    """Run Boltz-2 affinity head directly on fixed input coordinates."""
    pad_token_mask = batch["token_pad_mask"][0]
    rec_mask = (batch["mol_type"][0] == 0) * pad_token_mask
    lig_mask = batch["affinity_token_mask"][0].to(torch.bool) * pad_token_mask
    cross_pair_mask = (
        lig_mask[:, None] * rec_mask[None, :]
        + rec_mask[:, None] * lig_mask[None, :]
        + lig_mask[:, None] * lig_mask[None, :]
    )
    z_affinity = out["z"] * cross_pair_mask[None, :, :, None]

    coords_affinity = batch["coords"].detach()
    if coords_affinity.ndim == 3:
        coords_affinity = coords_affinity[:, None]

    s_inputs = model.input_embedder(batch, affinity=True)

    with torch.autocast("cuda", enabled=False):
        if getattr(model, "affinity_ensemble", False):
            d1 = model.affinity_module1(
                s_inputs=s_inputs.detach(),
                z=z_affinity.detach(),
                x_pred=coords_affinity,
                feats=batch,
                multiplicity=1,
                use_kernels=model.use_kernels,
            )
            d1["affinity_probability_binary"] = torch.sigmoid(
                d1["affinity_logits_binary"]
            )
            d2 = model.affinity_module2(
                s_inputs=s_inputs.detach(),
                z=z_affinity.detach(),
                x_pred=coords_affinity,
                feats=batch,
                multiplicity=1,
                use_kernels=model.use_kernels,
            )
            d2["affinity_probability_binary"] = torch.sigmoid(
                d2["affinity_logits_binary"]
            )
            aff_out = {
                "affinity_pred_value": (
                    d1["affinity_pred_value"] + d2["affinity_pred_value"]
                )
                / 2,
                "affinity_probability_binary": (
                    d1["affinity_probability_binary"]
                    + d2["affinity_probability_binary"]
                )
                / 2,
                "affinity_pred_value1": d1["affinity_pred_value"],
                "affinity_probability_binary1": d1["affinity_probability_binary"],
                "affinity_pred_value2": d2["affinity_pred_value"],
                "affinity_probability_binary2": d2["affinity_probability_binary"],
            }
            if getattr(model, "affinity_mw_correction", False):
                model_coef = 1.03525938
                mw_coef = -0.59992683
                bias = 2.83288489
                mw = batch["affinity_mw"][0] ** 0.3
                aff_out["affinity_pred_value"] = (
                    model_coef * aff_out["affinity_pred_value"]
                    + mw_coef * mw
                    + bias
                )
            return aff_out

        d0 = model.affinity_module(
            s_inputs=s_inputs.detach(),
            z=z_affinity.detach(),
            x_pred=coords_affinity,
            feats=batch,
            multiplicity=1,
            use_kernels=model.use_kernels,
        )
        return {
            "affinity_pred_value": d0["affinity_pred_value"],
            "affinity_probability_binary": torch.sigmoid(
                d0["affinity_logits_binary"]
            ),
        }


class _FixedPoseBoltz2(_BaseBoltz2):
    """Stock Boltz2 subclass that scores fixed input coordinates without diffusion."""

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Any]:
        if _HAS_BOLTZINA or not getattr(self, "skip_run_structure", False):
            return super().predict_step(batch, batch_idx, dataloader_idx)

        try:
            saved_aff = self.affinity_prediction
            saved_conf = self.confidence_prediction
            self.affinity_prediction = False
            self.confidence_prediction = False
            try:
                out = self(
                    batch,
                    recycling_steps=self.predict_args["recycling_steps"],
                    num_sampling_steps=self.predict_args["sampling_steps"],
                    diffusion_samples=self.predict_args["diffusion_samples"],
                    max_parallel_samples=self.predict_args["max_parallel_samples"],
                    run_confidence_sequentially=True,
                )
            finally:
                self.affinity_prediction = saved_aff
                self.confidence_prediction = saved_conf

            pred_dict: dict[str, Any] = {"exception": False}
            if "keys_dict_batch" in self.predict_args:
                for key in self.predict_args["keys_dict_batch"]:
                    pred_dict[key] = batch[key]
            pred_dict["masks"] = batch["atom_pad_mask"]
            pred_dict["token_masks"] = batch["token_pad_mask"]
            pred_dict["s"] = out["s"]
            pred_dict["z"] = out["z"]
            if saved_aff:
                pred_dict.update(_score_affinity_on_fixed_coords(self, batch, out))
            return pred_dict
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                return {"exception": True}
            raise


def _resolve_cache_dir(cache_dir: Optional[Union[str, Path]]) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    return Path(get_cache_path())


def _auto_use_kernels() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        if torch.cuda.get_device_properties(0).major < 8:
            return False
        return (
            importlib.util.find_spec("cuequivariance_ops_torch") is not None
            or importlib.util.find_spec("cuequivariance_ops_cu12") is not None
        )
    except Exception:  # noqa: BLE001
        return False


def load_boltz2_model(
    affinity_checkpoint: Optional[Union[str, Path]] = None,
    sampling_steps_affinity: int = 200,
    diffusion_samples_affinity: int = 5,
    subsample_msa: bool = True,
    num_subsampled_msa: int = 1024,
    model: str = "boltz2",
    step_scale: Optional[float] = None,
    affinity_mw_correction: bool = False,
    skip_run_structure: bool = True,
    confidence_prediction: bool = False,
    use_kernels: Optional[bool] = None,
    run_trunk_and_structure: bool = True,
    predict_affinity_args: Optional[dict[str, Any]] = None,
    pairformer_args: Optional[dict[str, Any]] = None,
    msa_args: Optional[dict[str, Any]] = None,
    steering_args: Optional[dict[str, Any]] = None,
    diffusion_process_args: Optional[dict[str, Any]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Any:
    """Load and return a Boltz-2 model configured for fixed-pose affinity scoring."""
    resolved_cache = _resolve_cache_dir(cache_dir)

    ckpt = (
        Path(affinity_checkpoint)
        if affinity_checkpoint is not None
        else resolved_cache / "boltz2_aff.ckpt"
    )
    if not ckpt.exists():
        raise FileNotFoundError(f"Affinity checkpoint not found at {ckpt}")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    resolved_use_kernels = (
        _auto_use_kernels() if use_kernels is None else bool(use_kernels)
    )

    if predict_affinity_args is None:
        predict_affinity_args = {
            "recycling_steps": 5,
            "sampling_steps": sampling_steps_affinity,
            "diffusion_samples": diffusion_samples_affinity,
            "max_parallel_samples": 1,
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        }

    diffusion_params = (
        asdict(Boltz2DiffusionParams())
        if diffusion_process_args is None
        else diffusion_process_args
    )
    diffusion_params["step_scale"] = 1.5 if step_scale is None else step_scale

    if pairformer_args is None:
        pairformer_args = asdict(PairformerArgsV2())

    if msa_args is None:
        msa_args = asdict(
            MSAModuleArgs(
                subsample_msa=subsample_msa,
                num_subsampled_msa=num_subsampled_msa,
                use_paired_feature=model == "boltz2",
            )
        )

    if steering_args is None:
        steering_args = {
            "fk_steering": False,
            "physical_guidance_update": False,
            "contact_guidance_update": False,
            "guidance_update": False,
        }

    model_cls = _BaseBoltz2 if _HAS_BOLTZINA else _FixedPoseBoltz2
    target_dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        from abcfold.boltz.boltz_worker import _fast_load_boltz_checkpoint

        model_module = _fast_load_boltz_checkpoint(
            torch,
            model_cls,
            ckpt,
            device=target_dev,
            strict=True,
            predict_args=predict_affinity_args,
            diffusion_process_args=diffusion_params,
            ema=False,
            pairformer_args=pairformer_args,
            msa_args=msa_args,
            steering_args=steering_args,
            affinity_mw_correction=affinity_mw_correction,
            skip_run_structure=skip_run_structure,
            run_trunk_and_structure=run_trunk_and_structure,
            confidence_prediction=True,
            use_kernels=resolved_use_kernels,
        )
    except Exception:  # noqa: BLE001
        model_module = model_cls.load_from_checkpoint(
            ckpt,
            strict=True,
            predict_args=predict_affinity_args,
            map_location=target_dev,
            diffusion_process_args=diffusion_params,
            ema=False,
            pairformer_args=pairformer_args,
            msa_args=msa_args,
            steering_args=steering_args,
            affinity_mw_correction=affinity_mw_correction,
            skip_run_structure=skip_run_structure,
            run_trunk_and_structure=run_trunk_and_structure,
            confidence_prediction=True,
            use_kernels=resolved_use_kernels,
        )
    model_module.confidence_prediction = confidence_prediction
    model_module.eval()
    return model_module


def predict_affinity(
    out_dir: Union[str, Path],
    model_module: Any = None,
    output_dir: Optional[Union[str, Path]] = None,
    structures_dir: Optional[Union[str, Path]] = None,
    msa_dir: Optional[Union[str, Path]] = None,
    constraints_dir: Optional[Union[str, Path]] = None,
    template_dir: Optional[Union[str, Path]] = None,
    extra_mols_dir: Optional[Union[str, Path]] = None,
    manifest_path: Optional[Union[str, Path]] = None,
    affinity_checkpoint: Optional[Union[str, Path]] = None,
    sampling_steps_affinity: int = 200,
    diffusion_samples_affinity: int = 5,
    subsample_msa: bool = True,
    num_subsampled_msa: int = 1024,
    model: str = "boltz2",
    step_scale: Optional[float] = None,
    override: bool = False,
    num_workers: int = 1,
    strategy: str = "auto",
    accelerator: Optional[str] = None,
    devices: int = 1,
    affinity_mw_correction: bool = False,
    seed: Optional[int] = None,
    batch_size: int = 1,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Any:
    """Run Boltz-2 affinity inference over prepared pre_affinity npz poses."""
    del override  # kept for signature compatibility with upstream Boltzina
    out_dir = Path(out_dir)
    resolved_cache = _resolve_cache_dir(cache_dir)
    mol_dir = resolved_cache / "mols"

    if seed is not None:
        seed_everything(seed)

    if model_module is None:
        model_module = load_boltz2_model(
            affinity_checkpoint=affinity_checkpoint,
            sampling_steps_affinity=sampling_steps_affinity,
            diffusion_samples_affinity=diffusion_samples_affinity,
            subsample_msa=subsample_msa,
            num_subsampled_msa=num_subsampled_msa,
            model=model,
            step_scale=step_scale,
            affinity_mw_correction=affinity_mw_correction,
            cache_dir=resolved_cache,
        )

    structures_path = (
        out_dir / "processed" / "structures"
        if structures_dir is None
        else Path(structures_dir)
    )
    msa_path = out_dir / "processed" / "msa" if msa_dir is None else Path(msa_dir)
    constraints_path = (
        out_dir / "processed" / "constraints"
        if constraints_dir is None
        else Path(constraints_dir)
    )
    template_path = (
        out_dir / "processed" / "templates"
        if template_dir is None
        else Path(template_dir)
    )
    manifest = Manifest.load(
        out_dir / "processed" / "manifest.json"
        if manifest_path is None
        else Path(manifest_path)
    )
    predictions_path = (
        out_dir / "predictions" if output_dir is None else Path(output_dir)
    )
    extra_mols_path = (
        predictions_path / "mols"
        if extra_mols_dir is None
        else Path(extra_mols_dir)
    )

    pred_writer = BoltzAffinityWriter(
        data_dir=structures_path,
        output_dir=predictions_path,
    )

    eff_num_workers = 0 if len(manifest.records) <= 2 else num_workers
    dm_kwargs: dict[str, Any] = {
        "manifest": manifest,
        "target_dir": predictions_path,
        "msa_dir": msa_path,
        "mol_dir": mol_dir,
        "num_workers": eff_num_workers,
        "constraints_dir": constraints_path,
        "template_dir": template_path,
        "extra_mols_dir": extra_mols_path,
        "override_method": "other",
        "affinity": True,
    }
    if "batch_size" in inspect.signature(Boltz2InferenceDataModule.__init__).parameters:
        dm_kwargs["batch_size"] = batch_size
    data_module = Boltz2InferenceDataModule(**dm_kwargs)

    resolved_accelerator = (
        accelerator
        if accelerator is not None
        else ("gpu" if torch.cuda.is_available() else "cpu")
    )
    precision: Union[int, str] = (
        "bf16-mixed"
        if (model != "boltz1" and resolved_accelerator == "gpu")
        else 32
    )

    trainer = Trainer(
        default_root_dir=out_dir,
        strategy=strategy,
        callbacks=[pred_writer],
        accelerator=resolved_accelerator,
        devices=devices,
        precision=precision,
    )
    return trainer.predict(
        model_module,
        datamodule=data_module,
        return_predictions=True,
    )

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union

from abcfold.gpus import to_physical
from abcfold.openfold3.af3_to_openfold3 import OpenfoldJson
from abcfold.openfold3.check_install import (ensure_openfold_checkpoint,
                                             ensure_openfold_env,
                                             openfold_cache_dir)

logger = logging.getLogger("logger")


def run_openfold(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    config: dict,
    save_input: bool = False,
    test: bool = False,
    number_of_models: int = 5,
    template_hits_path: Path | None = None,
    input_ckpt: Optional[Union[str, Path]] = None,
    num_recycles: Optional[int] = None,
    diffusion_steps: Optional[int] = None,
    fetch_templates: bool = False,
    template_mirror_dir: Optional[Union[str, Path]] = None,
    gpus: str = "all",
) -> bool:
    """
    Run OpenFold 3 using the input JSON file.

    Template structures are never fetched from RCSB unless ``fetch_templates``
    is set; ``template_mirror_dir`` points OpenFold3 at a local mmCIF mirror.

    Returns:
        bool: True if the run produced at least one structure.
    """
    input_json = Path(input_json)
    output_dir = Path(output_dir)

    env = None
    if not test:
        logger.debug("Checking OpenFold3 environment")
        env = ensure_openfold_env(config=config)

    ckpt_path: Optional[Path] = None
    if input_ckpt is not None:
        ckpt_path = Path(input_ckpt)
        if not ckpt_path.exists():
            logger.error("OpenFold3 checkpoint not found: %s", ckpt_path)
            return False
        logger.info("Using user provided OpenFold3 checkpoint: %s", ckpt_path)

    cache_dir = openfold_cache_dir(config)
    ckpt_name = config.get("openfold_checkpoint")
    if ckpt_path is None and env is not None:
        ensure_openfold_checkpoint(env, cache_dir, ckpt_name)

    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir)
        if save_input:
            logger.info("Saving input json file and msa to the output directory")
            working_dir = output_dir
            working_dir.mkdir(parents=True, exist_ok=True)

        openfold_json = OpenfoldJson(working_dir, templates=template_hits_path)
        openfold_json.json_to_json(input_json)

        out_file = working_dir.joinpath(f"{input_json.stem}.json")
        openfold_json.write_json(out_file)

        runner_yaml = working_dir / "openfold3_runner.yml"
        openfold_json.write_yaml(
            runner_yaml,
            num_recycles=num_recycles,
            diffusion_steps=diffusion_steps,
            fetch_templates=fetch_templates,
            template_mirror_dir=template_mirror_dir,
            cache_dir=cache_dir,
        )

        logger.info("Running OpenFold 3 using seeds: %s", openfold_json.seeds)
        openfold_out_dir = output_dir / "openfold_results"
        cmd = (
            generate_openfold_command(
                out_file,
                openfold_out_dir,
                runner_yaml,
                ckpt_path,
                number_of_models,
                checkpoint_name=None if ckpt_path else ckpt_name,
            )
            if not test
            else generate_openfold_test_command()
        )
        if not test:
            extra_env = _gpu_env(gpus) | {"OPENFOLD_CACHE": str(cache_dir)}
            try:
                assert env is not None
                env.run(cmd, extra_env=extra_env)
            except subprocess.CalledProcessError as e:
                _write_error(output_dir, "openfold_error.log", e)
                return False

            if not list(openfold_out_dir.rglob("*_model.cif")):
                logger.error(
                    "OpenFold 3 exited without writing any structures"
                )
                return False

    logger.info("OpenFold 3 run complete")
    logger.info("Output files are in %s", output_dir)
    return True


def _gpu_env(gpus: str) -> dict:
    if gpus in (None, "all"):
        return {}
    if gpus == "cpu":
        return {"CUDA_VISIBLE_DEVICES": ""}
    return {"CUDA_VISIBLE_DEVICES": to_physical(gpus)}


def _write_error(output_dir: Path, name: str, e: subprocess.CalledProcessError):
    stderr = e.stderr or ""
    output_dir.mkdir(parents=True, exist_ok=True)
    err_file = output_dir / name
    err_file.write_text(stderr)
    logger.error("OpenFold 3 run failed. Error log is in %s", err_file)


def generate_openfold_command(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    runner_yaml: Union[str, Path],
    ckpt_path: Optional[Union[str, Path]],
    number_of_models: int = 5,
    checkpoint_name: Optional[str] = None,
) -> list:
    """Build the ``run_openfold predict`` command."""
    cmd = [
        "run_openfold",
        "predict",
        "--query_json", str(input_json),
        "--runner_yaml", str(runner_yaml),
        "--num_diffusion_samples", str(number_of_models),
        "--output_dir", str(output_dir),
        "--use_msa_server", "false",
    ]
    if ckpt_path is not None:
        cmd += ["--inference_ckpt_path", str(ckpt_path)]
    elif checkpoint_name:
        cmd += ["--inference_ckpt_name", str(checkpoint_name)]
    return cmd


def generate_openfold_test_command() -> list:
    return [
        "run_openfold",
        "predict",
        "--help",
    ]

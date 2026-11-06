import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union

from abcfold.gpus import to_physical
from abcfold.msa.guard import (ShallowMsaError, check_backend_n_msa,
                               check_protenix_input, min_msa_rows,
                               msa_requested)
from abcfold.protenix.af3_to_protenix import ProtenixJson
from abcfold.protenix.check_install import ensure_protenix_env

logger = logging.getLogger("logger")

DEFAULT_PROTENIX_MODEL = "protenix_base_default_v1.0.0"


def cuda_toolkit_available(env=None, environ=None) -> bool:
    """True when nvcc is reachable, i.e. Protenix can JIT-build its fused
    layernorm kernel. Checks CUDA_HOME first, then PATH (inside ``env``)."""
    environ = os.environ if environ is None else environ
    cuda_home = environ.get("CUDA_HOME") or environ.get("CUDA_PATH")
    if cuda_home and Path(cuda_home, "bin", "nvcc").exists():
        return True
    if env is not None and hasattr(env, "has_executable"):
        return env.has_executable("nvcc")
    return shutil.which("nvcc") is not None


def protenix_runtime_env(
    config: dict,
    gpus: str = "all",
    nvcc_available: bool = False,
    environ=None,
) -> dict:
    """Environment variables for a Protenix run."""
    environ = os.environ if environ is None else environ
    extra = {}
    if "LAYERNORM_TYPE" not in environ and not nvcc_available:
        # Without nvcc Protenix fails while JIT-compiling the fused layernorm
        extra["LAYERNORM_TYPE"] = "torch"
    root = config.get("protenix_root_dir")
    if root:
        extra["PROTENIX_ROOT_DIR"] = str(Path(str(root)).expanduser())
    if gpus not in (None, "all"):
        physical = to_physical(gpus, environ)
        if physical is not None:
            extra["CUDA_VISIBLE_DEVICES"] = physical
    return extra


def run_protenix(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    config: dict,
    save_input: bool = False,
    test: bool = False,
    number_of_models: int = 5,
    num_recycles: int = 10,
    diffusion_steps: Optional[int] = None,
    gpus: str = "all",
) -> bool:
    """
    Run Protenix using the input JSON file.

    Returns:
        bool: True only if every seed produced at least one structure.
    """
    input_json = Path(input_json)
    output_dir = Path(output_dir)

    env = None
    extra_env: dict = {}
    if not test:
        logger.debug("Checking Protenix environment")
        env = ensure_protenix_env(config=config)
        extra_env = protenix_runtime_env(
            config, gpus=gpus, nvcc_available=cuda_toolkit_available(env)
        )
        if extra_env.get("LAYERNORM_TYPE") == "torch":
            logger.info("nvcc not found; running Protenix with LAYERNORM_TYPE=torch")

    min_rows = min_msa_rows(config)
    input_params = json.loads(input_json.read_text())
    msa_expected = any(
        msa_requested(s["protein"]) for s in input_params.get("sequences", [])
        if "protein" in s
    )

    # The MSA files referenced by unpairedMsaPath/pairedMsaPath live in
    # working_dir, which must outlive every Protenix process below.
    with tempfile.TemporaryDirectory(prefix="abcfold-protenix-") as temp_dir:
        working_dir = Path(temp_dir)
        if save_input:
            logger.info("Saving input json file and msa to the output directory")
            working_dir = output_dir

        protenix_json = ProtenixJson(working_dir)
        protenix_json.json_to_json(input_json)

        for seed in protenix_json.seeds:
            out_file = working_dir.joinpath(f"{input_json.stem}_seed-{seed}.json")

            protenix_json.write_json(out_file)
            if msa_expected:
                check_protenix_input(json.loads(out_file.read_text()), min_rows)
            protenix_out_dir = output_dir / f"protenix_results_seed-{seed}"
            logger.info("Running Protenix using seed: %s", seed)
            cmd = (
                generate_protenix_command(
                    out_file,
                    protenix_out_dir,
                    config,
                    number_of_models,
                    num_recycles,
                    seed=seed,
                    diffusion_steps=diffusion_steps,
                )
                if not test
                else generate_protenix_test_command()
            )

            if test:
                logger.info("Skipping Protenix backend execution in test mode")
                continue
            log_path = output_dir / f"protenix_seed-{seed}.log"
            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                assert env is not None
                env.run(cmd, extra_env=extra_env, log_file=log_path)
            except subprocess.CalledProcessError as e:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_err_file = output_dir / "protenix_error.log"
                output_err_file.write_text(e.stderr or "")
                logger.error(
                    "Protenix run failed. Error log is in %s", output_err_file
                )
                return False

            if msa_expected and log_path.exists():
                try:
                    check_backend_n_msa("Protenix", log_path.read_text(), min_rows)
                except ShallowMsaError as exc:
                    (output_dir / "protenix_error.log").write_text(str(exc))
                    logger.error(str(exc))
                    raise

            # Protenix logs per-sample exceptions and still exits 0
            if not list(protenix_out_dir.rglob("*.cif")):
                logger.error(
                    "Protenix exited without writing a structure for seed %s", seed
                )
                return False

    logger.info("Protenix run complete")
    logger.info("Output files are in %s", output_dir)
    return True


def generate_protenix_command(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    config: dict,
    number_of_models: int,
    num_recycles: int,
    seed: int,
    diffusion_steps: Optional[int] = None,
) -> list:
    """Build the ``runner.inference`` command for Protenix."""
    use_msa = False
    with open(str(input_json), "r") as f:
        data = json.load(f)
    for entry in data[0].get("sequences", []):
        chain = entry.get("proteinChain")
        if chain and (
            "msa" in chain or chain.get("unpairedMsaPath") or chain.get("pairedMsaPath")
        ):
            use_msa = True
            break

    model = config.get("protenix_model") or DEFAULT_PROTENIX_MODEL
    cmd = [
        "python",
        "-m",
        "runner.inference",
        "--model_name",
        str(model),
        "--input_json_path",
        str(input_json),
        "--dump_dir",
        str(output_dir),
        "--sample_diffusion.N_sample",
        str(number_of_models),
        "--model.N_cycle",
        str(num_recycles),
        "--seeds",
        str(seed),
        "--use_msa",
        str(use_msa),
        "--need_atom_confidence",
        "True",
    ]
    if diffusion_steps is not None:
        cmd += ["--sample_diffusion.N_step", str(diffusion_steps)]
    return cmd


def generate_protenix_test_command() -> list:
    return [
        "protenix",
        "predict",
        "--help",
    ]

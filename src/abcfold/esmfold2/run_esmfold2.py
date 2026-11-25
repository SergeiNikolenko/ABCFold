import json
import logging
import subprocess
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

from abcfold.esmfold2.af3_to_esmfold2 import Esmfold2json
from abcfold.esmfold2.check_install import ensure_esmfold2_env
from abcfold.gpus import cuda_env
from abcfold.output.cofolding_common import input_dir_for

logger = logging.getLogger("logger")

RUNNER = Path(__file__).parent.joinpath("runner_template.py")
# The code default is 200 while the published example passes 100, so the value
# is always sent explicitly.
NUM_SAMPLING_STEPS = 100
NUM_LOOPS = 20
DEFAULT_ESMFOLD2_MODEL = "biohub/ESMFold2"


def run_esmfold2(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    config: dict,
    save_input: bool = False,
    test: bool = False,
    number_of_models: int = 5,
    gpus: Optional[str] = "all",
) -> bool:
    """
    Run ESMFold2 using the input JSON file

    ESMFold2 has no command line interface, so abcfold runs a small script
    inside the isolated environment that drives the Python API.

    Args:
        input_json (Union[str, Path]): Path to the input JSON file
        output_dir (Union[str, Path]): Path to the output directory
        config (dict): Configuration dictionary
        save_input (bool): If True, save the input JSON file and MSA to the output
        directory
        test (bool): If True, run the test command
        number_of_models (int): Number of models to generate
        gpus (Optional[str]): ``--gpus`` selection, mapped through the inherited
        CUDA_VISIBLE_DEVICES mask

    Returns:
        Bool: True if the ESMFold2 run was successful, False otherwise

    Raises:
        subprocess.CalledProcessError: If the ESMFold2 command returns an error
    """
    input_json = Path(input_json)
    output_dir = Path(output_dir)

    logger.debug("Checking if ESMFold2 is installed")
    env = ensure_esmfold2_env(config=config)

    if test:
        try:
            env.run(generate_esmfold2_test_command())
        except subprocess.CalledProcessError:
            logger.error("ESMFold2 run failed")
            return False
        logger.info("ESMFold2 run complete")
        return True

    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir)
        if save_input:
            logger.info("Saving msa to the output directory")
            working_dir = input_dir_for(output_dir, "esmfold2")
            working_dir.mkdir(parents=True, exist_ok=True)

        esmfold2_json = Esmfold2json(working_dir)
        esmfold2_json.json_to_json(input_json)

        out_file = working_dir.joinpath(f"{input_json.stem}_esmfold2.json")
        esmfold2_json.write_json(out_file)

        options_file = working_dir.joinpath(f"{input_json.stem}_esmfold2_options.json")
        options_file.write_text(
            json.dumps(
                {
                    "model_repo": (config.get("esmfold2_model")
                                   or DEFAULT_ESMFOLD2_MODEL),
                    "device": config.get("esmfold2_device") or "cuda",
                    "num_loops": NUM_LOOPS,
                    "num_sampling_steps": NUM_SAMPLING_STEPS,
                    "num_diffusion_samples": number_of_models,
                }
            )
        )

        esmfold2_out_dir = output_dir / "esmfold2_results"
        esmfold2_out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Running ESMFold2 using seeds: %s", esmfold2_json.seeds)

        extra_env = cuda_env(gpus)
        weights = config.get("esmfold2_weights")
        if weights:
            # ESMFold2 pulls its weights from Hugging Face. Pointing the cache
            # at one directory keeps the ~10 GB download out of every new
            # environment and lets several hosts share it.
            extra_env["HF_HOME"] = str(Path(str(weights)).expanduser())

        try:
            env.run(
                generate_esmfold2_command(out_file, esmfold2_out_dir, options_file),
                extra_env=extra_env,
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr or ""
            if stderr:
                # The working directory is temporary unless --save_input is
                # set, so the log goes next to the results where it survives.
                output_dir.mkdir(parents=True, exist_ok=True)
                output_err_file = output_dir / "esmfold2_error.log"
                output_err_file.write_text(stderr)
                logger.error(
                    "ESMFold2 run failed. Error log is in %s", output_err_file
                )
            else:
                logger.error("ESMFold2 run failed")
            return False

    logger.info("ESMFold2 run complete")
    logger.info("Output files are in %s", output_dir)
    return True


def _gpu_slots(gpus: Optional[str]) -> list[Optional[str]]:
    if gpus in (None, "all", "cpu"):
        return [gpus]
    slots: list[Optional[str]] = [
        g.strip() for g in str(gpus).split(",") if g.strip()
    ]
    return slots or ["all"]


def run_esmfold2_batch(
    items: Iterable[Tuple[Union[str, Path], Union[str, Path]]],
    config: dict,
    save_input: bool = False,
    test: bool = False,
    number_of_models: int = 5,
    gpus: Optional[str] = "all",
) -> dict[Path, bool]:
    """Fold many jobs loading the EsmFold2Model weights once per GPU."""
    item_list = [(Path(inp), Path(out)) for inp, out in items]
    if not item_list:
        return {}
    if test:
        return {out: True for _, out in item_list}

    env = ensure_esmfold2_env(config=config)
    gpu_slots = _gpu_slots(gpus)
    results: dict[Path, bool] = {out: False for _, out in item_list}

    with tempfile.TemporaryDirectory(prefix="abcfold-esmfold2-batch-") as temp_dir:
        work_root = Path(temp_dir)
        runner_copy = work_root / "esmfold2_runner.py"
        shutil.copyfile(RUNNER, runner_copy)

        prepared_jobs = []
        for idx, (inp, job_out) in enumerate(item_list):
            job_out.mkdir(parents=True, exist_ok=True)
            working_dir = (
                input_dir_for(job_out, "esmfold2")
                if save_input
                else work_root / f"job_{idx}"
            )
            working_dir.mkdir(parents=True, exist_ok=True)
            esmfold2_json = Esmfold2json(working_dir)
            esmfold2_json.json_to_json(inp)
            desc_file = working_dir / f"{inp.stem}_esmfold2.json"
            esmfold2_json.write_json(desc_file)
            esm_out = job_out / "esmfold2_results"
            esm_out.mkdir(parents=True, exist_ok=True)
            prepared_jobs.append((job_out, desc_file, esm_out))

        base_options = {
            "model_repo": config.get("esmfold2_model") or DEFAULT_ESMFOLD2_MODEL,
            "device": config.get("esmfold2_device") or "cuda",
            "num_loops": NUM_LOOPS,
            "num_sampling_steps": NUM_SAMPLING_STEPS,
            "num_diffusion_samples": number_of_models,
        }

        n_shards = min(len(gpu_slots), len(prepared_jobs))
        shards: list[list[tuple[Path, Path, Path]]] = [[] for _ in range(n_shards)]
        for idx, job_entry in enumerate(prepared_jobs):
            shards[idx % n_shards].append(job_entry)

        def _run_shard(shard_idx: int) -> None:
            shard_jobs = shards[shard_idx]
            manifest_path = work_root / f"manifest_{shard_idx}.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "options": base_options,
                        "jobs": [
                            {"description": str(desc), "output_dir": str(esm_out)}
                            for _, desc, esm_out in shard_jobs
                        ],
                    },
                    indent=2,
                )
            )
            extra_env = cuda_env(gpu_slots[shard_idx])
            weights = config.get("esmfold2_weights")
            if weights:
                extra_env["HF_HOME"] = str(Path(str(weights)).expanduser())
            cmd = ["python", str(runner_copy), "--manifest", str(manifest_path)]
            try:
                env.run(cmd, extra_env=extra_env)
            except subprocess.CalledProcessError as exc:
                for job_out, _, _ in shard_jobs:
                    if exc.stderr:
                        (job_out / "esmfold2_error.log").write_text(exc.stderr)
                return
            for job_out, _, esm_out in shard_jobs:
                results[job_out] = bool(list(esm_out.rglob("*_model.cif")))

        if n_shards > 1:
            with ThreadPoolExecutor(max_workers=n_shards) as pool:
                futs = [pool.submit(_run_shard, i) for i in range(n_shards)]
                for fut in as_completed(futs):
                    fut.result()
        else:
            _run_shard(0)

    return results


def generate_esmfold2_command(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    options_file: Union[str, Path],
) -> list:
    """
    Generate the ESMFold2 command

    Args:
        input_json (Union[str, Path]): Path to the converted input description
        output_dir (Union[str, Path]): Path to the output directory
        options_file (Union[str, Path]): Path to the runtime options file

    Returns:
        list: The ESMFold2 command
    """
    return [
        "python",
        str(RUNNER),
        str(input_json),
        str(output_dir),
        str(options_file),
    ]


def generate_esmfold2_test_command() -> list:
    """
    Generate the test command for ESMFold2

    ESMFold2 has no CLI to interrogate, so the import is the smoke test.

    Args:
        None

    Returns:
        list: The ESMFold2 test command
    """

    return [
        "python",
        "-c",
        "from esm.models.esmfold2 import EsmFold2Model, ESMFold2InputBuilder; "
        "print('esmfold2 ok')",
    ]

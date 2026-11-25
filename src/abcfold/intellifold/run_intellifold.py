import json
import logging
import queue
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

from abcfold.backend_envs import jax_cache_dir, jax_runtime_env
from abcfold.gpus import cuda_env, detect_gpu_slots
from abcfold.intellifold.af3_to_intellifold import Intellifoldjson
from abcfold.intellifold.check_install import ensure_intellifold_env
from abcfold.output.cofolding_common import input_dir_for

logger = logging.getLogger("logger")


def run_intellifold(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    config: dict,
    save_input: bool = False,
    test: bool = False,
    number_of_models: int = 5,
    num_recycles: int = 10,
    gpus: Optional[str] = "all",
) -> bool:
    """
    Run IntelliFold using the input JSON file

    Args:
        input_json (Union[str, Path]): Path to the input JSON file
        output_dir (Union[str, Path]): Path to the output directory
        config (dict): Configuration dictionary
        save_input (bool): If True, save the input JSON file and MSA to the output
        directory
        test (bool): If True, run the test command
        number_of_models (int): Number of models to generate
        num_recycles (int): Number of trunk recycles
        gpus (Optional[str]): ``--gpus`` selection, mapped through the inherited
        CUDA_VISIBLE_DEVICES mask

    Returns:
        Bool: True if the IntelliFold run was successful, False otherwise

    Raises:
        subprocess.CalledProcessError: If the IntelliFold command returns an error
    """
    input_json = Path(input_json)
    output_dir = Path(output_dir)

    logger.debug("Checking if IntelliFold is installed")
    env = ensure_intellifold_env(config=config)

    intellifold_weight_dir = config.get("intellifold_weights")
    if intellifold_weight_dir is not None and intellifold_weight_dir != "None":
        model_dir = Path(intellifold_weight_dir)
    else:
        model_dir = Path.home().joinpath(".intellifold", "model_v2")
    # The CLI default is ./model_v2, relative to the working directory, so an
    # absolute path is always passed. Weights are fetched on first use.
    model_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir)
        if save_input:
            logger.info("Saving msa to the output directory")
            working_dir = input_dir_for(output_dir, "intellifold")
            working_dir.mkdir(parents=True, exist_ok=True)

        intellifold_json = Intellifoldjson(working_dir)
        intellifold_json.json_to_json(input_json)

        out_file = working_dir.joinpath(f"{input_json.stem}_intellifold.json")
        intellifold_json.write_json(out_file)

        intellifold_out_dir = output_dir / "intellifold_results"
        logger.info("Running IntelliFold using seeds: %s", intellifold_json.seeds)
        cache_dir = jax_cache_dir(config)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cmd = (
            generate_intellifold_command(
                out_file,
                intellifold_out_dir,
                model_dir,
                number_of_models,
                num_recycles=num_recycles,
                jax_compilation_cache_dir=cache_dir,
            )
            if not test
            else generate_intellifold_test_command()
        )

        seeds = [int(s) for s in intellifold_json.seeds] or [42]
        gpu_slots = detect_gpu_slots(gpus)
        shard_seeds = (
            not test
            and len(seeds) > 1
            and len(gpu_slots) > 1
            and gpu_slots != ["cpu"]
            and gpu_slots != ["all"]
        )
        if not shard_seeds:
            extra_env = {**jax_runtime_env(config), **cuda_env(gpus)}
            try:
                env.run(cmd, extra_env=extra_env)
            except subprocess.CalledProcessError as e:
                stderr = e.stderr or ""
                if stderr:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_err_file = output_dir / "intellifold_error.log"
                    output_err_file.write_text(stderr)
                    logger.error(
                        "IntelliFold run failed. Error log is in %s", output_err_file
                    )
                else:
                    logger.error("IntelliFold run failed")
                return False
        else:
            n_shards = min(len(gpu_slots), len(seeds))
            shards: list[list[int]] = [[] for _ in range(n_shards)]
            for idx, s in enumerate(seeds):
                shards[idx % n_shards].append(s)

            def _run_seed_shard(shard_idx: int) -> bool:
                shard_dict = dict(intellifold_json.intellifold_dict)
                shard_dict["modelSeeds"] = shards[shard_idx]
                shard_json = working_dir / f"{input_json.stem}_shard{shard_idx}.json"
                shard_json.write_text(json.dumps(shard_dict, indent=2))
                s_cmd = generate_intellifold_command(
                    shard_json,
                    intellifold_out_dir,
                    model_dir,
                    number_of_models,
                    num_recycles=num_recycles,
                    jax_compilation_cache_dir=cache_dir,
                )
                s_env = {**jax_runtime_env(config), **cuda_env(gpu_slots[shard_idx])}
                try:
                    env.run(s_cmd, extra_env=s_env)
                except subprocess.CalledProcessError as e:
                    if e.stderr:
                        output_dir.mkdir(parents=True, exist_ok=True)
                        (output_dir / "intellifold_error.log").write_text(e.stderr)
                    logger.error("IntelliFold shard %d failed", shard_idx)
                    return False
                return True

            with ThreadPoolExecutor(max_workers=n_shards) as pool:
                futs = [pool.submit(_run_seed_shard, i) for i in range(n_shards)]
                if not all(f.result() for f in as_completed(futs)):
                    return False

    logger.info("IntelliFold run complete")
    logger.info("Output files are in %s", output_dir)
    return True


def run_intellifold_batch(
    items: Iterable[Tuple[Union[str, Path], Union[str, Path]]],
    config: dict,
    save_input: bool = False,
    test: bool = False,
    number_of_models: int = 5,
    num_recycles: int = 10,
    gpus: Optional[str] = "all",
) -> dict[Path, bool]:
    """Run many IntelliFold jobs across all visible GPUs sharing a warm XLA cache."""
    item_list = [(Path(inp), Path(out)) for inp, out in items]
    if not item_list:
        return {}
    if test:
        return {out: True for _, out in item_list}

    gpu_slots = detect_gpu_slots(gpus)
    slot_q: "queue.Queue[str]" = queue.Queue()
    for s in gpu_slots:
        slot_q.put(s)
    results: dict[Path, bool] = {out: False for _, out in item_list}

    def _run_one(inp: Path, job_out: Path) -> None:
        slot = slot_q.get()
        try:
            results[job_out] = run_intellifold(
                input_json=inp,
                output_dir=job_out,
                config=config,
                save_input=save_input,
                test=False,
                number_of_models=number_of_models,
                num_recycles=num_recycles,
                gpus=slot,
            )
        finally:
            slot_q.put(slot)

    n_workers = min(len(gpu_slots), len(item_list))
    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_run_one, inp, out) for inp, out in item_list]
            for fut in as_completed(futs):
                fut.result()
    else:
        for inp, out in item_list:
            _run_one(inp, out)
    return results


def generate_intellifold_command(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    model_dir: Union[str, Path],
    number_of_models: int = 5,
    num_recycles: int = 10,
    jax_compilation_cache_dir: Optional[Union[str, Path]] = None,
) -> list:
    """
    Generate the IntelliFold command

    Seeds are not a command line flag: the vendored AlphaFold 3 engine reads
    them from ``modelSeeds`` in the input JSON and runs once per seed.
    Everything after the bare ``--`` is forwarded verbatim to the engine's
    absl flags. The engine's data pipeline stays off: it would need the
    AlphaFold 3 databases, and ABCFold has already resolved the alignments.

    Args:
        input_json (Union[str, Path]): Path to the input JSON file
        output_dir (Union[str, Path]): Path to the output directory
        model_dir (Union[str, Path]): Path to the model weights directory
        number_of_models (int): Number of diffusion samples to generate
        num_recycles (int): Number of trunk recycles

    Returns:
        list: The IntelliFold command
    """
    return [
        "intellifold",
        "predict",
        str(input_json),
        f"--model-dir={model_dir}",
        f"--output-dir={output_dir}",
        "--",
        f"--num_diffusion_samples={number_of_models}",
        f"--num_recycles={num_recycles}",
        # Without this the engine appends a timestamp when the output
        # directory already exists, which breaks output collection.
        "--force_output_dir",
        "--norun_data_pipeline",
        "--resolve_msa_overlaps=false",
        *(
            [f"--jax_compilation_cache_dir={jax_compilation_cache_dir}"]
            if jax_compilation_cache_dir
            else []
        ),
    ]


def generate_intellifold_test_command() -> list:
    """
    Generate the test command for IntelliFold

    Args:
        None

    Returns:
        list: The IntelliFold test command
    """

    return [
        "intellifold",
        "predict",
        "--help",
    ]

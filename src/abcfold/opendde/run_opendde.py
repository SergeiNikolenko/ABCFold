import json
import logging
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

from abcfold.gpus import (cuda_env, detect_gpu_slots, fast_torch_env,
                          plan_gpu_workers)
from abcfold.opendde.af3_to_opendde import OpenDDEjson
from abcfold.opendde.check_install import ensure_opendde_env
from abcfold.output.cofolding_common import input_dir_for

logger = logging.getLogger("logger")

DEFAULT_OPENDDE_MODEL = "opendde_v1"


def run_opendde(
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
    Run OpenDDE using the input JSON file

    Args:
        input_json (Union[str, Path]): Path to the input JSON file
        output_dir (Union[str, Path]): Path to the output directory
        config (dict): Configuration dictionary
        save_input (bool): If True, save the input JSON file and MSA to the output
        directory
        test (bool): If True, run the test command
        number_of_models (int): Number of models to generate
        num_recycles (int): Number of Pairformer cycles
        gpus (Optional[str]): ``--gpus`` selection, mapped through the inherited
        CUDA_VISIBLE_DEVICES mask

    Returns:
        Bool: True if the OpenDDE run was successful, False otherwise

    Raises:
        subprocess.CalledProcessError: If the OpenDDE command returns an error
    """
    input_json = Path(input_json)
    output_dir = Path(output_dir)

    logger.debug("Checking if OpenDDE is installed")
    env = ensure_opendde_env(config=config)

    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir)
        if save_input:
            logger.info("Saving msa to the output directory")
            working_dir = input_dir_for(output_dir, "opendde")
            working_dir.mkdir(parents=True, exist_ok=True)

        opendde_json = OpenDDEjson(working_dir)
        opendde_json.json_to_json(input_json)

        out_file = working_dir.joinpath(f"{input_json.stem}_opendde.json")
        opendde_json.write_json(out_file)

        opendde_out_dir = output_dir / "opendde_results"
        logger.info("Running OpenDDE using seeds: %s", opendde_json.seeds)
        cmd = (
            generate_opendde_command(
                out_file,
                opendde_out_dir,
                config,
                number_of_models,
                opendde_json.seeds,
                use_msa=opendde_json.uses_msa,
                num_recycles=num_recycles,
            )
            if not test
            else generate_opendde_test_command()
        )

        seeds = [int(s) for s in opendde_json.seeds] or [42]
        gpu_slots = detect_gpu_slots(gpus)
        shard_seeds = (
            not test
            and len(seeds) > 1
            and len(gpu_slots) > 1
            and gpu_slots != ["cpu"]
            and gpu_slots != ["all"]
        )
        root_dir = config.get("opendde_root_dir")

        def _make_env(slot: Optional[str]) -> dict:
            e = {**fast_torch_env(), **cuda_env(slot)}
            if root_dir:
                e["OPENDDE_ROOT_DIR"] = str(Path(str(root_dir)).expanduser())
            return e

        if not shard_seeds:
            try:
                env.run(cmd, extra_env=_make_env(gpus))
            except subprocess.CalledProcessError as e:
                stderr = e.stderr or ""
                if stderr:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_err_file = output_dir / "opendde_error.log"
                    output_err_file.write_text(stderr)
                    logger.error(
                        "OpenDDE run failed. Error log is in %s", output_err_file
                    )
                else:
                    logger.error("OpenDDE run failed")
                return False
        else:
            n_shards = min(len(gpu_slots), len(seeds))
            shards: list[list[int]] = [[] for _ in range(n_shards)]
            for idx, s in enumerate(seeds):
                shards[idx % n_shards].append(s)

            def _run_seed_shard(shard_idx: int) -> bool:
                s_cmd = generate_opendde_command(
                    out_file,
                    opendde_out_dir,
                    config,
                    number_of_models,
                    shards[shard_idx],
                    use_msa=opendde_json.uses_msa,
                    num_recycles=num_recycles,
                )
                try:
                    env.run(s_cmd, extra_env=_make_env(gpu_slots[shard_idx]))
                except subprocess.CalledProcessError as e:
                    if e.stderr:
                        output_dir.mkdir(parents=True, exist_ok=True)
                        (output_dir / "opendde_error.log").write_text(e.stderr)
                    logger.error("OpenDDE shard %d failed", shard_idx)
                    return False
                return True

            with ThreadPoolExecutor(max_workers=n_shards) as pool:
                futs = [pool.submit(_run_seed_shard, i) for i in range(n_shards)]
                if not all(f.result() for f in as_completed(futs)):
                    return False

    logger.info("OpenDDE run complete")
    logger.info("Output files are in %s", output_dir)
    return True


def _gpu_slot_list(gpus: Optional[str]) -> list[Optional[str]]:
    return list(detect_gpu_slots(gpus))


def run_opendde_batch(
    items: Iterable[Tuple[Union[str, Path], Union[str, Path]]],
    config: dict,
    save_input: bool = False,
    test: bool = False,
    number_of_models: int = 5,
    num_recycles: int = 10,
    gpus: Optional[str] = "all",
    workers_per_gpu: Optional[int] = None,
) -> dict[Path, bool]:
    """Run many OpenDDE jobs in one invocation per GPU (weights loaded once)."""
    item_list = [(Path(inp), Path(out)) for inp, out in items]
    if not item_list:
        return {}
    if test:
        return {out: True for _, out in item_list}

    env = ensure_opendde_env(config=config)
    gpu_slots: list[Optional[str]] = list(
        plan_gpu_workers(
            gpus,
            len(item_list),
            engine="opendde",
            workers_per_gpu=workers_per_gpu,
        )
    )
    results: dict[Path, bool] = {out: False for _, out in item_list}

    with tempfile.TemporaryDirectory(prefix="abcfold-opendde-batch-") as temp_dir:
        work_root = Path(temp_dir)
        shared_msa_dir = work_root / "msas"
        shared_msa_dir.mkdir(parents=True, exist_ok=True)
        groups: dict[tuple[tuple[int, ...], bool], list[tuple[Path, dict]]] = {}
        for idx, (inp, job_out) in enumerate(item_list):
            job_out.mkdir(parents=True, exist_ok=True)
            working_dir = (
                input_dir_for(job_out, "opendde")
                if save_input
                else shared_msa_dir
            )
            working_dir.mkdir(parents=True, exist_ok=True)
            oj = OpenDDEjson(working_dir)
            entry = oj.json_to_json(inp)
            key = (tuple(int(s) for s in (oj.seeds or [42])), bool(oj.uses_msa))
            groups.setdefault(key, []).append((job_out, entry))

        def _run_shard(
            shard_idx: int,
            seeds_key: tuple[int, ...],
            use_msa: bool,
            shard_items: list[tuple[Path, dict]],
            gpu_slot: Optional[str],
        ) -> None:
            shard_dir = work_root / f"shard_{shard_idx}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            batch_json = shard_dir / "opendde_batch.json"
            batch_json.write_text(json.dumps([it[1] for it in shard_items], indent=2))
            batch_out = shard_dir / "opendde_results"
            cmd = generate_opendde_command(
                batch_json,
                batch_out,
                config,
                number_of_models,
                list(seeds_key),
                use_msa=use_msa,
                num_recycles=num_recycles,
            )
            extra_env = {**fast_torch_env(), **cuda_env(gpu_slot)}
            root_dir = config.get("opendde_root_dir")
            if root_dir:
                extra_env["OPENDDE_ROOT_DIR"] = str(Path(str(root_dir)).expanduser())
            try:
                env.run(cmd, extra_env=extra_env)
            except subprocess.CalledProcessError as exc:
                for job_out, _ in shard_items:
                    if exc.stderr:
                        (job_out / "opendde_error.log").write_text(exc.stderr)
                return
            for job_out, entry in shard_items:
                job_name = entry["name"]
                src = batch_out / job_name
                dst_root = job_out / "opendde_results"
                dst = dst_root / job_name
                if src.exists() and list(src.rglob("*.cif")):
                    if dst.exists():
                        shutil.rmtree(dst)
                    dst_root.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
                    results[job_out] = True

        tasks = []
        shard_counter = 0
        for (seeds_key, use_msa), group_items in groups.items():
            n_shards = min(len(gpu_slots), len(group_items))
            shards: list[list[tuple[Path, dict]]] = [[] for _ in range(n_shards)]
            for idx, item in enumerate(group_items):
                shards[idx % n_shards].append(item)
            for slot_idx, shard_items in enumerate(shards):
                tasks.append(
                    (
                        shard_counter,
                        seeds_key,
                        use_msa,
                        shard_items,
                        gpu_slots[slot_idx],
                    )
                )
                shard_counter += 1

        if len(tasks) > 1 and len(gpu_slots) > 1:
            with ThreadPoolExecutor(
                max_workers=min(len(gpu_slots), len(tasks))
            ) as pool:
                futs = [pool.submit(_run_shard, *t) for t in tasks]
                for fut in as_completed(futs):
                    fut.result()
        else:
            for t in tasks:
                _run_shard(*t)

    return results


def generate_opendde_command(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    config: dict,
    number_of_models: int = 5,
    seeds: Union[list, None] = None,
    use_msa: bool = False,
    num_recycles: int = 10,
    use_tfg_guidance: bool = False,
) -> list:
    """
    Generate the OpenDDE command

    Args:
        input_json (Union[str, Path]): Path to the input JSON file
        output_dir (Union[str, Path]): Path to the output directory
        config (dict): Configuration dictionary
        number_of_models (int): Number of samples to generate per seed
        seeds (list): Random seeds to use for the OpenDDE run
        use_msa (bool): If True, the converted input carries MSA paths
        num_recycles (int): Number of Pairformer cycles

    Returns:
        list: The OpenDDE command
    """
    seeds = seeds if seeds else [42]

    # Booleans are click bools and need an explicit value. --use_msa defaults to
    # true, which would start a ColabFold search when the input carries no
    # alignment, so it is always passed explicitly.
    cmd = [
        "opendde",
        "pred",
        "-i",
        str(input_json),
        "-o",
        str(output_dir),
        "-n",
        config.get("opendde_model") or DEFAULT_OPENDDE_MODEL,
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--sample",
        str(number_of_models),
        "--cycle",
        str(num_recycles),
        "--use_msa",
        "true" if use_msa else "false",
        "--use_template",
        "false",
        "--use_rna_msa",
        "false",
    ]
    if use_tfg_guidance:
        cmd += ["--use_tfg_guidance", "true"]

    return cmd


def generate_opendde_test_command() -> list:
    """
    Generate the test command for OpenDDE

    Args:
        None

    Returns:
        list: The OpenDDE test command
    """

    return [
        "opendde",
        "pred",
        "--help",
    ]

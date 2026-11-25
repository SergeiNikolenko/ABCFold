import json
import logging
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

from abcfold.gpus import (detect_gpu_slots, fast_torch_env, plan_gpu_workers,
                          resolve_openfold_low_mem, to_physical)
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

        seeds = [int(s) for s in openfold_json.seeds] or [42]
        gpu_slots = detect_gpu_slots(gpus)
        shard_seeds = (
            not test
            and len(seeds) > 1
            and len(gpu_slots) > 1
            and gpu_slots != ["cpu"]
            and gpu_slots != ["all"]
        )
        low_mem = resolve_openfold_low_mem(gpus)
        runner_yaml = working_dir / "openfold3_runner.yml"
        openfold_json.write_yaml(
            runner_yaml,
            num_recycles=num_recycles,
            diffusion_steps=diffusion_steps,
            fetch_templates=fetch_templates,
            template_mirror_dir=template_mirror_dir,
            cache_dir=cache_dir,
            low_mem=low_mem,
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
            if not shard_seeds:
                extra_env = (
                    fast_torch_env()
                    | _gpu_env(gpus)
                    | {"OPENFOLD_CACHE": str(cache_dir)}
                )
                try:
                    assert env is not None
                    env.run(cmd, extra_env=extra_env)
                except subprocess.CalledProcessError as e:
                    _write_error(output_dir, "openfold_error.log", e)
                    return False
            else:
                assert env is not None
                n_shards = min(len(gpu_slots), len(seeds))
                shards: list[list[int]] = [[] for _ in range(n_shards)]
                for idx, s in enumerate(seeds):
                    shards[idx % n_shards].append(s)

                def _run_seed_shard(shard_idx: int) -> bool:
                    s_seeds = shards[shard_idx]
                    s_yaml = working_dir / f"openfold3_runner_shard{shard_idx}.yml"
                    s_oj = OpenfoldJson(working_dir, create_files=False)
                    s_oj.seeds = s_seeds
                    s_oj.templates = openfold_json.templates
                    s_oj._use_templates = openfold_json._use_templates
                    s_oj.write_yaml(
                        s_yaml,
                        num_recycles=num_recycles,
                        diffusion_steps=diffusion_steps,
                        fetch_templates=fetch_templates,
                        template_mirror_dir=template_mirror_dir,
                        cache_dir=cache_dir,
                        low_mem=resolve_openfold_low_mem(gpu_slots[shard_idx]),
                    )
                    s_yaml.write_text(
                        s_oj.write_yaml(
                            s_yaml,
                            num_recycles=num_recycles,
                            diffusion_steps=diffusion_steps,
                            fetch_templates=fetch_templates,
                            template_mirror_dir=template_mirror_dir,
                            cache_dir=cache_dir,
                            low_mem=resolve_openfold_low_mem(gpu_slots[shard_idx]),
                        )
                    )
                    s_cmd = generate_openfold_command(
                        out_file,
                        openfold_out_dir,
                        s_yaml,
                        ckpt_path,
                        number_of_models,
                        checkpoint_name=None if ckpt_path else ckpt_name,
                    )
                    s_env = (
                        fast_torch_env()
                        | _gpu_env(gpu_slots[shard_idx])
                        | {"OPENFOLD_CACHE": str(cache_dir)}
                    )
                    try:
                        env.run(s_cmd, extra_env=s_env)
                    except subprocess.CalledProcessError as e:
                        _write_error(output_dir, "openfold_error.log", e)
                        return False
                    return True

                with ThreadPoolExecutor(max_workers=n_shards) as pool:
                    futs = [pool.submit(_run_seed_shard, i) for i in range(n_shards)]
                    if not all(f.result() for f in as_completed(futs)):
                        return False

            if not list(openfold_out_dir.rglob("*_model.cif")):
                logger.error(
                    "OpenFold 3 exited without writing any structures"
                )
                return False

    logger.info("OpenFold 3 run complete")
    logger.info("Output files are in %s", output_dir)
    return True


def _gpu_slot_list(gpus: str) -> list[str]:
    return detect_gpu_slots(gpus)


def run_openfold_batch(
    items: Iterable[Tuple[Union[str, Path], Union[str, Path]]],
    config: dict,
    save_input: bool = False,
    test: bool = False,
    number_of_models: int = 5,
    input_ckpt: Optional[Union[str, Path]] = None,
    num_recycles: Optional[int] = None,
    diffusion_steps: Optional[int] = None,
    fetch_templates: bool = False,
    template_mirror_dir: Optional[Union[str, Path]] = None,
    gpus: str = "all",
    workers_per_gpu: Optional[int] = None,
) -> dict[Path, bool]:
    """Run many OpenFold3 queries in one invocation per GPU (weights loaded once)."""
    item_list = [(Path(inp), Path(out)) for inp, out in items]
    if not item_list:
        return {}
    if test:
        return {out: True for _, out in item_list}

    env = ensure_openfold_env(config=config)
    ckpt_path: Optional[Path] = None
    if input_ckpt is not None:
        ckpt_path = Path(input_ckpt)
        if not ckpt_path.exists():
            logger.error("OpenFold3 checkpoint not found: %s", ckpt_path)
            return {out: False for _, out in item_list}
    cache_dir = openfold_cache_dir(config)
    ckpt_name = config.get("openfold_checkpoint")
    if ckpt_path is None:
        ensure_openfold_checkpoint(env, cache_dir, ckpt_name)

    gpu_slots = plan_gpu_workers(
        gpus,
        len(item_list),
        engine="openfold3",
        workers_per_gpu=workers_per_gpu,
    )
    results: dict[Path, bool] = {out: False for _, out in item_list}

    with tempfile.TemporaryDirectory(prefix="abcfold-openfold3-batch-") as temp_dir:
        work_root = Path(temp_dir)
        shared_msa_dir = work_root / "msas"
        shared_msa_dir.mkdir(parents=True, exist_ok=True)
        groups: dict[tuple[int, ...], list[tuple[Path, str, dict, OpenfoldJson]]] = {}
        for idx, (inp, job_out) in enumerate(item_list):
            job_out.mkdir(parents=True, exist_ok=True)
            working_dir = job_out if save_input else shared_msa_dir
            working_dir.mkdir(parents=True, exist_ok=True)
            oj = OpenfoldJson(working_dir)
            oj.json_to_json(inp)
            query_name = next(iter(oj.openfold_dict["queries"]))
            seeds_key = tuple(int(s) for s in (oj.seeds or [42]))
            groups.setdefault(seeds_key, []).append(
                (job_out, query_name, oj.openfold_dict["queries"][query_name], oj)
            )

        def _run_shard(
            shard_idx: int,
            seeds_key: tuple[int, ...],
            shard_items: list[tuple[Path, str, dict, OpenfoldJson]],
            gpu_slot: str,
        ) -> None:
            shard_dir = work_root / f"shard_{shard_idx}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            merged_queries = {}
            for _, qname, qbody, _ in shard_items:
                merged_queries[qname] = qbody
            query_file = shard_dir / "queries.json"
            query_file.write_text(
                json.dumps(
                    {"seeds": list(seeds_key), "queries": merged_queries}, indent=2
                )
            )
            runner_yaml = shard_dir / "openfold3_runner.yml"
            shard_items[0][3].write_yaml(
                runner_yaml,
                num_recycles=num_recycles,
                diffusion_steps=diffusion_steps,
                fetch_templates=fetch_templates,
                template_mirror_dir=template_mirror_dir,
                cache_dir=cache_dir,
                low_mem=resolve_openfold_low_mem(gpu_slot),
            )
            batch_out = shard_dir / "openfold_results"
            cmd = generate_openfold_command(
                query_file,
                batch_out,
                runner_yaml,
                ckpt_path,
                number_of_models,
                checkpoint_name=None if ckpt_path else ckpt_name,
            )
            extra_env = (
                fast_torch_env()
                | _gpu_env(gpu_slot)
                | {"OPENFOLD_CACHE": str(cache_dir)}
            )
            try:
                env.run(cmd, extra_env=extra_env)
            except subprocess.CalledProcessError as exc:
                for job_out, _, _, _ in shard_items:
                    _write_error(job_out, "openfold_error.log", exc)
                return

            for job_out, qname, _, _ in shard_items:
                src = batch_out / qname
                dst_root = job_out / "openfold_results"
                dst = dst_root / qname
                if src.exists() and list(src.rglob("*_model.cif")):
                    if dst.exists():
                        shutil.rmtree(dst)
                    dst_root.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
                    results[job_out] = True

        tasks = []
        shard_counter = 0
        for seeds_key, group_items in groups.items():
            n_shards = min(len(gpu_slots), len(group_items))
            shards: list[list[tuple[Path, str, dict, OpenfoldJson]]] = [
                [] for _ in range(n_shards)
            ]
            for idx, item in enumerate(group_items):
                shards[idx % n_shards].append(item)
            for slot_idx, shard_items in enumerate(shards):
                tasks.append(
                    (shard_counter, seeds_key, shard_items, gpu_slots[slot_idx])
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

import json
import logging
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

from abcfold.gpus import (detect_gpu_slots, fast_torch_env, plan_gpu_workers,
                          to_physical)
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
    extra = fast_torch_env(environ)
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

        seeds = [int(s) for s in protenix_json.seeds] or [42]
        out_file = working_dir.joinpath(f"{input_json.stem}_protenix.json")
        protenix_json.write_json(out_file)
        if msa_expected:
            check_protenix_input(json.loads(out_file.read_text()), min_rows)
        protenix_out_dir = output_dir / "protenix_results"
        logger.info("Running Protenix using seeds: %s", seeds)
        cmd = (
            generate_protenix_command(
                out_file,
                protenix_out_dir,
                config,
                number_of_models,
                num_recycles,
                seed=seeds,
                diffusion_steps=diffusion_steps,
            )
            if not test
            else generate_protenix_test_command()
        )

        if test:
            logger.info("Skipping Protenix backend execution in test mode")
            return True

        output_dir.mkdir(parents=True, exist_ok=True)
        gpu_slots = detect_gpu_slots(gpus)
        shard_seeds = (
            len(seeds) > 1
            and len(gpu_slots) > 1
            and gpu_slots != ["cpu"]
            and gpu_slots != ["all"]
        )
        if not shard_seeds:
            log_path = output_dir / "protenix.log"
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
        else:
            assert env is not None
            nvcc_ok = cuda_toolkit_available(env)
            n_shards = min(len(gpu_slots), len(seeds))
            shards: list[list[int]] = [[] for _ in range(n_shards)]
            for idx, s in enumerate(seeds):
                shards[idx % n_shards].append(s)

            def _run_seed_shard(shard_idx: int) -> bool:
                s_list = shards[shard_idx]
                s_cmd = generate_protenix_command(
                    out_file,
                    protenix_out_dir,
                    config,
                    number_of_models,
                    num_recycles,
                    seed=s_list,
                    diffusion_steps=diffusion_steps,
                )
                s_env = protenix_runtime_env(
                    config, gpus=gpu_slots[shard_idx], nvcc_available=nvcc_ok
                )
                s_log = output_dir / f"protenix_shard{shard_idx}.log"
                try:
                    env.run(s_cmd, extra_env=s_env, log_file=s_log)
                except subprocess.CalledProcessError as e:
                    (output_dir / "protenix_error.log").write_text(e.stderr or "")
                    logger.error("Protenix shard %d failed", shard_idx)
                    return False
                if msa_expected and s_log.exists():
                    check_backend_n_msa("Protenix", s_log.read_text(), min_rows)
                return True

            with ThreadPoolExecutor(max_workers=n_shards) as pool:
                futs = [pool.submit(_run_seed_shard, i) for i in range(n_shards)]
                if not all(f.result() for f in as_completed(futs)):
                    return False

        # Protenix logs per-sample exceptions and still exits 0
        all_cifs = list(protenix_out_dir.rglob("*.cif"))
        if not all_cifs:
            logger.error("Protenix exited without writing any structures")
            return False
        for seed in seeds:
            if not any(f"seed_{seed}" in cif.parts for cif in all_cifs):
                if len(seeds) > 1:
                    logger.error(
                        "Protenix exited without writing a structure for seed %s", seed
                    )
                    return False

    logger.info("Protenix run complete")
    logger.info("Output files are in %s", output_dir)
    return True


def _gpu_slot_list(gpus: str) -> list[str]:
    return detect_gpu_slots(gpus)


def run_protenix_batch(
    items: Iterable[Tuple[Union[str, Path], Union[str, Path]]],
    config: dict,
    save_input: bool = False,
    test: bool = False,
    number_of_models: int = 5,
    num_recycles: int = 10,
    diffusion_steps: Optional[int] = None,
    gpus: str = "all",
    workers_per_gpu: Optional[int] = None,
) -> dict[Path, bool]:
    """Run many Protenix jobs loading weights once per GPU shard."""
    item_list = [(Path(inp), Path(out)) for inp, out in items]
    if not item_list:
        return {}
    if test:
        return {out: True for _, out in item_list}

    env = ensure_protenix_env(config=config)
    nvcc_ok = cuda_toolkit_available(env)
    min_rows = min_msa_rows(config)
    gpu_slots = plan_gpu_workers(
        gpus,
        len(item_list),
        engine="protenix",
        workers_per_gpu=workers_per_gpu,
    )

    groups: dict[tuple[int, ...], list[tuple[Path, Path, dict, bool]]] = {}
    with tempfile.TemporaryDirectory(prefix="abcfold-protenix-batch-") as temp_dir:
        work_root = Path(temp_dir)
        shared_msa_dir = work_root / "msas"
        shared_msa_dir.mkdir(parents=True, exist_ok=True)
        for inp, job_out in item_list:
            job_out.mkdir(parents=True, exist_ok=True)
            working_dir = job_out if save_input else shared_msa_dir
            working_dir.mkdir(parents=True, exist_ok=True)
            raw_params = json.loads(inp.read_text())
            msa_exp = any(
                msa_requested(s["protein"])
                for s in raw_params.get("sequences", [])
                if "protein" in s
            )
            pj = ProtenixJson(working_dir)
            entry = pj.json_to_json(inp)
            if msa_exp:
                check_protenix_input([entry], min_rows)
            seeds_key = tuple(int(s) for s in (pj.seeds or [42]))
            groups.setdefault(seeds_key, []).append((inp, job_out, entry, msa_exp))

        results: dict[Path, bool] = {out: False for _, out in item_list}

        def _run_shard(
            shard_idx: int,
            seeds_key: tuple[int, ...],
            shard_items: list[tuple[Path, Path, dict, bool]],
            gpu_slot: str,
        ) -> None:
            shard_dir = work_root / f"shard_{shard_idx}_{'_'.join(map(str, seeds_key))}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            batch_json = shard_dir / "protenix_batch.json"
            batch_json.write_text(json.dumps([it[2] for it in shard_items], indent=2))
            dump_dir = shard_dir / "dump"
            log_path = shard_dir / "protenix_batch.log"
            cmd = generate_protenix_command(
                batch_json,
                dump_dir,
                config,
                number_of_models,
                num_recycles,
                seed=list(seeds_key),
                diffusion_steps=diffusion_steps,
            )
            extra_env = protenix_runtime_env(
                config, gpus=gpu_slot, nvcc_available=nvcc_ok
            )
            try:
                env.run(cmd, extra_env=extra_env, log_file=log_path)
            except subprocess.CalledProcessError as exc:
                for _, job_out, _, _ in shard_items:
                    (job_out / "protenix_error.log").write_text(exc.stderr or "")
                return

            for _, job_out, entry, _ in shard_items:
                job_name = entry["name"]
                src_dir = dump_dir / job_name
                dst_root = job_out / "protenix_results"
                dst_dir = dst_root / job_name
                if src_dir.exists() and list(src_dir.rglob("*.cif")):
                    if dst_dir.exists():
                        shutil.rmtree(dst_dir)
                    dst_root.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src_dir), str(dst_dir))
                    results[job_out] = True

        tasks: list[
            tuple[int, tuple[int, ...], list[tuple[Path, Path, dict, bool]], str]
        ] = []
        shard_counter = 0
        for seeds_key, group_items in groups.items():
            n_shards = min(len(gpu_slots), len(group_items))
            shards: list[list[tuple[Path, Path, dict, bool]]] = [
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


def generate_protenix_command(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    config: dict,
    number_of_models: int,
    num_recycles: int,
    seed: Union[int, Sequence[int]],
    diffusion_steps: Optional[int] = None,
) -> list:
    """Build the runner.inference command for Protenix."""
    use_msa = False
    with open(str(input_json), "r") as f:
        data = json.load(f)
    for job in data:
        for entry in job.get("sequences", []):
            chain = entry.get("proteinChain")
            if chain and (
                "msa" in chain
                or chain.get("unpairedMsaPath")
                or chain.get("pairedMsaPath")
            ):
                use_msa = True
                break
        if use_msa:
            break

    seeds_arg = (
        ",".join(str(int(s)) for s in seed)
        if isinstance(seed, (list, tuple))
        else str(seed)
    )
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
        seeds_arg,
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
        "pred",
        "--help",
    ]

import json
import logging
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Union

from abcfold.gpus import (detect_gpu_slots, fast_torch_env, plan_gpu_workers,
                          to_physical)
from abcfold.boltz.af3_to_boltz import BoltzYaml
from abcfold.boltz.check_install import ensure_boltz_env
from abcfold.boltz.crystal_modes import apply_crystal_mode

logger = logging.getLogger("logger")
BOLTZ_WORKER = Path(__file__).parent.joinpath("boltz_worker.py")


def normalize_gpus(gpus: str) -> str | None:
    if gpus == "cpu":
        return ""
    if gpus == "all":
        return None

    gpu_ids = []
    for gpu in gpus.split(","):
        gpu = gpu.strip()
        if not gpu.isdigit():
            raise ValueError(f"Invalid GPU ID: {gpu}")
        gpu_ids.append(gpu)
    return ",".join(gpu_ids)


def _normalize_optional_int(value: int | str | None, name: str) -> int | None:
    if value is None or value == "None":
        return None
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be greater than 0")
    return value


def _boltz_cache_path(config: dict) -> str:
    boltz_weight_dir = config["boltz_weights"]
    if boltz_weight_dir is not None and boltz_weight_dir != "None":
        return str(boltz_weight_dir)
    return str(Path.home().joinpath(".boltz"))


def _apply_cuda_visible_devices(cmd: list[str], cuda_devices: str | None) -> list[str]:
    if cuda_devices is None:
        return cmd
    if cuda_devices != "":
        cuda_devices = to_physical(cuda_devices)
    return ["env", f"CUDA_VISIBLE_DEVICES={cuda_devices}", *cmd]


def _run_command(env, cmd: list[str], output_dir: Path) -> str | None:
    try:
        extra_env = fast_torch_env()
        try:
            stdout = env.run(cmd, capture_output=True, extra_env=extra_env)
        except TypeError:
            stdout = env.run(cmd, capture_output=True)
        if stdout and "WARNING: ran out of memory" in stdout:
            logger.error("Boltz ran out of memory")
            return None
        return stdout
    except subprocess.CalledProcessError as e:
        stderr = e.stderr or ""
        if stderr:
            logger.error(stderr)
            output_err_file = output_dir / "boltz_error.log"
            output_err_file.write_text(stderr)
            logger.error("Boltz run failed. Error log is in %s", output_err_file)
        else:
            logger.error("Boltz run failed")
        raise


def _has_boltz_structures(output_dir: Path, input_stem: str) -> bool:
    results = output_dir / f"boltz_results_{input_stem}"
    return any(results.rglob("*_model_*.cif")) if results.exists() else False


def _load_boltz_input(
    input_json: Path,
    boltz_mode: str,
    boltz_crystal_structure: Union[str, Path, None],
    boltz_ligand_chain: str | None,
    boltz_template_chain_id: list[str] | None,
    boltz_template_id: list[str] | None,
    boltz_template_force: bool,
    boltz_template_threshold: float | None,
    boltz_pocket_radius: float,
    boltz_pocket_max_distance: float,
    boltz_pocket_force: bool,
) -> dict:
    with input_json.open("r") as handle:
        input_params = json.load(handle)
    return apply_crystal_mode(
        input_params,
        mode=boltz_mode,
        crystal_structure=(
            Path(boltz_crystal_structure)
            if boltz_crystal_structure is not None
            else None
        ),
        ligand_chain=boltz_ligand_chain,
        template_chain_id=boltz_template_chain_id,
        template_id=boltz_template_id,
        template_force=boltz_template_force,
        template_threshold=(
            2.0 if boltz_template_threshold is None else boltz_template_threshold
        ),
        pocket_radius=boltz_pocket_radius,
        pocket_max_distance=boltz_pocket_max_distance,
        pocket_force=boltz_pocket_force,
    )


def _case_id_from_input(input_json: Path) -> str:
    if input_json.stem.startswith("abc_"):
        return input_json.stem[4:]
    return input_json.stem


def _link_or_copy_input(input_json: Path, output_dir: Path) -> None:
    case_id = _case_id_from_input(input_json)
    case_dir = output_dir / "outputs" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    target = case_dir / f"abc_{case_id}.json"
    if target.exists() or target.is_symlink():
        return
    try:
        target.symlink_to(input_json.resolve())
    except OSError:
        shutil.copy2(input_json, target)


def _batch_gpu_slots(gpus: str) -> list[str | None]:
    cuda_devices = normalize_gpus(gpus)
    if cuda_devices is None:
        detected = detect_gpu_slots(gpus)
        if detected == ["cpu"]:
            return [""]
        if detected == ["all"]:
            return [None]
        return list(detected)
    if cuda_devices == "":
        return [""]
    return list(cuda_devices.split(","))


def _estimate_params_tokens(params: dict) -> int:
    total = 0
    for entry in params.get("sequences", []):
        if not isinstance(entry, dict):
            continue
        for key in ("protein", "rna", "dna"):
            if key in entry:
                seq = entry[key].get("sequence") or ""
                ids = entry[key].get("id")
                copies = len(ids) if isinstance(ids, list) else 1
                total += len(seq) * max(1, copies)
        if "ligand" in entry:
            total += 25
    return max(1, total)


def _seed_gpu(gpu_slots: list[str | None], index: int) -> str | None:
    return gpu_slots[index % len(gpu_slots)]


def run_boltz(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    config: dict,
    save_input: bool = False,
    test: bool = False,
    number_of_models: int = 5,
    num_recycles: int = 10,
    gpus: str = "all",
    boltz_mode: str = "default",
    boltz_crystal_structure: Union[str, Path, None] = None,
    boltz_ligand_chain: str | None = None,
    boltz_template_chain_id: list[str] | None = None,
    boltz_template_id: list[str] | None = None,
    boltz_template_force: bool = False,
    boltz_template_threshold: float | None = None,
    boltz_pocket_radius: float = 6.0,
    boltz_pocket_max_distance: float = 6.0,
    boltz_pocket_force: bool = False,
    preprocessing_threads: int | str | None = None,
    num_workers: int | str | None = None,
    max_parallel_samples: int | str | None = None,
    sampling_steps: int = 200,
    boltz_use_potentials: bool = False,
) -> bool:
    """
    Run Boltz using the input JSON file

    Args:
        input_json (Union[str, Path]): Path to the input JSON file
        output_dir (Union[str, Path]): Path to the output directory
        config (dict): Configuration dictionary
        save_input (bool): If True, save the input yaml file and MSA to the output
        directory
        test (bool): If True, run the test command
        number_of_models (int): Number of models to generate
        num_recycles (int): Number of recycles to use

    Returns:
        Bool: True if the Boltz run was successful, False otherwise

    Raises:
        subprocess.CalledProcessError: If the Boltz command returns an error


    """
    input_json = Path(input_json)
    output_dir = Path(output_dir)
    preprocessing_threads = _normalize_optional_int(
        preprocessing_threads,
        "preprocessing_threads",
    )
    num_workers = _normalize_optional_int(num_workers, "num_workers")
    max_parallel_samples = _normalize_optional_int(
        max_parallel_samples,
        "max_parallel_samples",
    )

    env = None
    if not test:
        logger.debug("Checking if boltz is installed")
        env = ensure_boltz_env(config=config)

    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir)
        if save_input:
            logger.info("Saving input yaml file and msa to the output directory")
            working_dir = output_dir

        inline_threshold = (
            boltz_template_threshold
            if boltz_template_threshold is not None
            else (2.0 if boltz_template_force else None)
        )
        boltz_yaml = BoltzYaml(working_dir, template_threshold=inline_threshold)
        input_params = _load_boltz_input(
            input_json,
            boltz_mode,
            boltz_crystal_structure,
            boltz_ligand_chain,
            boltz_template_chain_id,
            boltz_template_id,
            boltz_template_force,
            boltz_template_threshold,
            boltz_pocket_radius,
            boltz_pocket_max_distance,
            boltz_pocket_force,
        )
        boltz_yaml.json_to_yaml(input_params)

        seeds = [int(s) for s in boltz_yaml.seeds]
        if test:
            for seed in seeds:
                out_file = working_dir.joinpath(f"{input_json.stem}_seed-{seed}.yaml")
                boltz_yaml.write_yaml(out_file)
                generate_boltz_test_command()
                logger.info("Skipping Boltz backend execution in test mode")
        else:
            try:
                gpu_slots = _batch_gpu_slots(gpus)
            except ValueError as e:
                logger.error("Invalid GPU configuration: %s", e)
                return False

            yaml_path = working_dir.joinpath(f"{input_json.stem}.yaml")
            boltz_yaml.write_yaml(yaml_path)
            seed_out_dirs = {
                seed: output_dir / f"boltz_results_{input_json.stem}_seed-{seed}"
                for seed in seeds
            }
            logger.info("Running Boltz persistent worker for seeds: %s", seeds)
            try:
                ok = _run_boltz_workers_for_seeds(
                    env=env,
                    inputs_source=yaml_path,
                    work_root=working_dir,
                    output_dir=output_dir,
                    config=config,
                    seeds=seeds,
                    seed_out_dirs=seed_out_dirs,
                    gpu_slots=gpu_slots,
                    number_of_models=number_of_models,
                    num_recycles=num_recycles,
                    sampling_steps=sampling_steps,
                    preprocessing_threads=preprocessing_threads,
                    num_workers=num_workers,
                    max_parallel_samples=max_parallel_samples,
                    use_potentials=boltz_use_potentials,
                )
            except subprocess.CalledProcessError:
                return False
            if not ok:
                return False
            for seed in seeds:
                if not _has_boltz_structures(
                    output_dir, f"{input_json.stem}_seed-{seed}"
                ):
                    logger.error(
                        "Boltz exited without writing a structure for seed %s", seed
                    )
                    return False

    logger.info("Boltz run complete")
    logger.info("Output files are in %s", output_dir)
    return True


def run_boltz_batch(
    input_jsons: Iterable[Union[str, Path]],
    output_dir: Union[str, Path],
    config: dict,
    save_input: bool = True,
    test: bool = False,
    number_of_models: int = 5,
    num_recycles: int = 10,
    gpus: str = "all",
    boltz_mode: str = "default",
    boltz_crystal_structure: Union[str, Path, None] = None,
    boltz_ligand_chain: str | None = None,
    boltz_template_chain_id: list[str] | None = None,
    boltz_template_id: list[str] | None = None,
    boltz_template_force: bool = False,
    boltz_template_threshold: float | None = None,
    boltz_pocket_radius: float = 6.0,
    boltz_pocket_max_distance: float = 6.0,
    boltz_pocket_force: bool = False,
    preprocessing_threads: int | str | None = 2,
    num_workers: int | str | None = 2,
    max_parallel_samples: int | str | None = None,
    parallel_seeds: bool = True,
    sampling_steps: int = 200,
    boltz_use_potentials: bool = False,
    workers_per_gpu: int | None = None,
) -> bool:
    """
    Run Boltz over many AlphaFold3 JSON files using one Boltz invocation per seed.
    """
    input_paths = [Path(path) for path in input_jsons]
    if not input_paths:
        logger.error("No Boltz input JSON files were provided")
        return False

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_threads = _normalize_optional_int(
        preprocessing_threads,
        "preprocessing_threads",
    )
    num_workers = _normalize_optional_int(num_workers, "num_workers")
    max_parallel_samples = _normalize_optional_int(
        max_parallel_samples,
        "max_parallel_samples",
    )

    env = None
    if not test:
        logger.debug("Checking if boltz is installed")
        env = ensure_boltz_env(config=config)

    batch_root = output_dir / "boltz_batch"
    inline_threshold = (
        boltz_template_threshold
        if boltz_template_threshold is not None
        else (2.0 if boltz_template_force else None)
    )

    if test:
        for input_json in input_paths:
            input_params = _load_boltz_input(
                input_json,
                boltz_mode,
                boltz_crystal_structure,
                boltz_ligand_chain,
                boltz_template_chain_id,
                boltz_template_id,
                boltz_template_force,
                boltz_template_threshold,
                boltz_pocket_radius,
                boltz_pocket_max_distance,
                boltz_pocket_force,
            )
            seeds = input_params.get("modelSeeds", [42])
            if isinstance(seeds, int):
                seeds = [seeds]
            if save_input:
                _link_or_copy_input(input_json, output_dir)
            for seed in seeds:
                seed_dir = batch_root / f"seed-{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                boltz_yaml = BoltzYaml(seed_dir, template_threshold=inline_threshold)
                boltz_yaml.json_to_yaml(input_params)
                boltz_yaml.write_yaml(seed_dir / f"{input_json.stem}_seed-{seed}.yaml")
        generate_boltz_test_command()
        logger.info("Skipping Boltz batch backend execution in test mode")
        return True

    prepared: list[tuple[Path, dict, list[int]]] = []
    all_seeds_set: set[int] = set()
    for input_json in input_paths:
        input_params = _load_boltz_input(
            input_json,
            boltz_mode,
            boltz_crystal_structure,
            boltz_ligand_chain,
            boltz_template_chain_id,
            boltz_template_id,
            boltz_template_force,
            boltz_template_threshold,
            boltz_pocket_radius,
            boltz_pocket_max_distance,
            boltz_pocket_force,
        )
        seeds_raw = input_params.get("modelSeeds", [42])
        seeds_list = (
            [int(seeds_raw)]
            if isinstance(seeds_raw, int)
            else [int(s) for s in seeds_raw]
        )
        all_seeds_set.update(seeds_list)
        if save_input:
            _link_or_copy_input(input_json, output_dir)
        prepared.append((input_json, input_params, seeds_list))

    all_seeds = sorted(all_seeds_set)
    seed_out_dirs = {
        seed: batch_root / f"seed-{seed}" / f"boltz_results_seed-{seed}"
        for seed in all_seeds
    }
    for seed_out in seed_out_dirs.values():
        seed_out.parent.mkdir(parents=True, exist_ok=True)

    try:
        normalize_gpus(gpus)
        max_tokens = max(
            (_estimate_params_tokens(p) for _, p, _ in prepared), default=300
        )
        planned = plan_gpu_workers(
            gpus,
            len(prepared),
            engine="boltz",
            est_tokens=max_tokens,
            workers_per_gpu=workers_per_gpu,
        )
        gpu_slots: list[str | None] = [
            "" if s == "cpu" else (None if s == "all" else s) for s in planned
        ]
    except ValueError as e:
        logger.error("Invalid GPU configuration: %s", e)
        return False

    try:
        if len(gpu_slots) > 1 and len(prepared) > 1 and (
            not parallel_seeds or len(all_seeds) == 1 or len(prepared) >= len(gpu_slots)
        ):
            shards: list[list[tuple[Path, dict, list[int]]]] = [
                [] for _ in range(min(len(gpu_slots), len(prepared)))
            ]
            for idx, item in enumerate(prepared):
                shards[idx % len(shards)].append(item)
            shared_msas_dir = batch_root / "msas"
            shared_msas_dir.mkdir(parents=True, exist_ok=True)
            # Pre-write unique content-hashed MSAs once before launching shards.
            pre_yamls: dict[Path, str] = {}
            for input_json, input_params, _ in prepared:
                boltz_yaml = BoltzYaml(
                    shared_msas_dir, template_threshold=inline_threshold
                )
                boltz_yaml.json_to_yaml(input_params)
                pre_yamls[input_json] = boltz_yaml.yaml_string

            def _run_input_shard(shard_idx: int) -> bool:
                shard_items = shards[shard_idx]
                shard_dir = batch_root / f"_shard_{shard_idx}"
                inputs_dir = shard_dir / "inputs"
                inputs_dir.mkdir(parents=True, exist_ok=True)
                shard_seed_records: dict[int, list[str]] = {}
                for input_json, input_params, seeds_list in shard_items:
                    yaml_out = inputs_dir / f"{input_json.stem}.yaml"
                    yaml_out.write_text(pre_yamls[input_json])
                    for seed in seeds_list:
                        shard_seed_records.setdefault(seed, []).append(input_json.stem)
                shard_seeds = sorted(shard_seed_records)
                return _run_single_boltz_worker(
                    env=env,
                    inputs_source=inputs_dir,
                    work_root=shard_dir,
                    output_dir=output_dir,
                    config=config,
                    seeds=shard_seeds,
                    seed_out_dirs=seed_out_dirs,
                    cuda_device=gpu_slots[shard_idx],
                    worker_tag=f"shard_{shard_idx}",
                    number_of_models=number_of_models,
                    num_recycles=num_recycles,
                    sampling_steps=sampling_steps,
                    preprocessing_threads=preprocessing_threads,
                    num_workers=num_workers,
                    max_parallel_samples=max_parallel_samples,
                    use_potentials=boltz_use_potentials,
                    seed_record_ids=shard_seed_records,
                )

            with ThreadPoolExecutor(max_workers=len(shards)) as pool:
                futures = [pool.submit(_run_input_shard, i) for i in range(len(shards))]
                if not all(f.result() for f in as_completed(futures)):
                    return False
        else:
            inputs_dir = batch_root / "inputs"
            msas_dir = batch_root / "msas"
            inputs_dir.mkdir(parents=True, exist_ok=True)
            msas_dir.mkdir(parents=True, exist_ok=True)
            seed_records: dict[int, list[str]] = {}
            for input_json, input_params, seeds_list in prepared:
                boltz_yaml = BoltzYaml(msas_dir, template_threshold=inline_threshold)
                boltz_yaml.json_to_yaml(input_params)
                boltz_yaml.write_yaml(inputs_dir / f"{input_json.stem}.yaml")
                for seed in seeds_list:
                    seed_records.setdefault(seed, []).append(input_json.stem)

            effective_slots = gpu_slots if parallel_seeds else gpu_slots[:1]
            ok = _run_boltz_workers_for_seeds(
                env=env,
                inputs_source=inputs_dir,
                work_root=batch_root,
                output_dir=output_dir,
                config=config,
                seeds=all_seeds,
                seed_out_dirs=seed_out_dirs,
                gpu_slots=effective_slots,
                number_of_models=number_of_models,
                num_recycles=num_recycles,
                sampling_steps=sampling_steps,
                preprocessing_threads=preprocessing_threads,
                num_workers=num_workers,
                max_parallel_samples=max_parallel_samples,
                use_potentials=boltz_use_potentials,
                seed_record_ids=seed_records,
            )
            if not ok:
                return False
    except subprocess.CalledProcessError:
        return False

    logger.info("Boltz batch run complete")
    logger.info("Output files are in %s", output_dir)
    return True


def _run_single_boltz_worker(
    *,
    env,
    inputs_source: Path,
    work_root: Path,
    output_dir: Path,
    config: dict,
    seeds: list[int],
    seed_out_dirs: dict[int, Path],
    cuda_device: str | None,
    worker_tag: str,
    number_of_models: int,
    num_recycles: int,
    sampling_steps: int,
    preprocessing_threads: int | None,
    num_workers: int | None,
    max_parallel_samples: int | None,
    use_potentials: bool,
    seed_record_ids: dict[int, list[str]] | None = None,
) -> bool:
    work_root.mkdir(parents=True, exist_ok=True)
    worker_copy = work_root / f"_boltz_worker_{worker_tag}.py"
    shutil.copyfile(BOLTZ_WORKER, worker_copy)
    shared_work_dir = work_root / f"_boltz_processed_{worker_tag}"
    manifest: dict[str, object] = {
        "cache": _boltz_cache_path(config),
        "shared_work_dir": str(shared_work_dir),
        "inputs_source": str(inputs_source),
        "seeds": [int(s) for s in seeds],
        "seed_out_dirs": {str(int(s)): str(seed_out_dirs[int(s)]) for s in seeds},
        "diffusion_samples": int(number_of_models),
        "recycling_steps": int(num_recycles),
        "sampling_steps": int(sampling_steps),
        "preprocessing_threads": preprocessing_threads,
        "num_workers": num_workers,
        "max_parallel_samples": max_parallel_samples,
        "use_potentials": bool(use_potentials),
    }
    if seed_record_ids is not None:
        manifest["seed_record_ids"] = {
            str(int(s)): list(seed_record_ids.get(int(s), [])) for s in seeds
        }
    manifest_path = work_root / f"_boltz_manifest_{worker_tag}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    cmd = generate_boltz_worker_command(worker_copy, manifest_path)
    cmd = _apply_cuda_visible_devices(cmd, cuda_device)
    stdout = _run_command(env, cmd, output_dir)
    return stdout is not None


def _run_boltz_workers_for_seeds(
    *,
    env,
    inputs_source: Path,
    work_root: Path,
    output_dir: Path,
    config: dict,
    seeds: list[int],
    seed_out_dirs: dict[int, Path],
    gpu_slots: list[str | None],
    number_of_models: int,
    num_recycles: int,
    sampling_steps: int,
    preprocessing_threads: int | None,
    num_workers: int | None,
    max_parallel_samples: int | None,
    use_potentials: bool,
    seed_record_ids: dict[int, list[str]] | None = None,
) -> bool:
    if len(gpu_slots) <= 1 or len(seeds) <= 1:
        return _run_single_boltz_worker(
            env=env,
            inputs_source=inputs_source,
            work_root=work_root,
            output_dir=output_dir,
            config=config,
            seeds=seeds,
            seed_out_dirs=seed_out_dirs,
            cuda_device=gpu_slots[0],
            worker_tag="gpu0",
            number_of_models=number_of_models,
            num_recycles=num_recycles,
            sampling_steps=sampling_steps,
            preprocessing_threads=preprocessing_threads,
            num_workers=num_workers,
            max_parallel_samples=max_parallel_samples,
            use_potentials=use_potentials,
            seed_record_ids=seed_record_ids,
        )

    seed_shards: list[list[int]] = [[] for _ in range(min(len(gpu_slots), len(seeds)))]
    for idx, seed in enumerate(seeds):
        seed_shards[idx % len(seed_shards)].append(seed)

    with ThreadPoolExecutor(max_workers=len(seed_shards)) as pool:
        futures = [
            pool.submit(
                _run_single_boltz_worker,
                env=env,
                inputs_source=inputs_source,
                work_root=work_root,
                output_dir=output_dir,
                config=config,
                seeds=shard,
                seed_out_dirs=seed_out_dirs,
                cuda_device=gpu_slots[shard_idx],
                worker_tag=f"gpu{shard_idx}",
                number_of_models=number_of_models,
                num_recycles=num_recycles,
                sampling_steps=sampling_steps,
                preprocessing_threads=preprocessing_threads,
                num_workers=num_workers,
                max_parallel_samples=max_parallel_samples,
                use_potentials=use_potentials,
                seed_record_ids=seed_record_ids,
            )
            for shard_idx, shard in enumerate(seed_shards)
        ]
        return all(f.result() for f in as_completed(futures))


def generate_boltz_worker_command(
    worker_script: Union[str, Path],
    manifest_path: Union[str, Path],
) -> list[str]:
    return ["python", str(worker_script), str(manifest_path)]


def generate_boltz_command(
    input_yaml: Union[str, Path],
    output_dir: Union[str, Path],
    config: dict,
    number_of_models: int = 5,
    num_recycles: int = 10,
    seed: int = 42,
    preprocessing_threads: int | str | None = None,
    num_workers: int | str | None = None,
    max_parallel_samples: int | str | None = None,
    sampling_steps: int = 200,
    boltz_use_potentials: bool = False,
) -> list:
    """
    Generate the Boltz command

    Args:
        input_yaml (Union[str, Path]): Path to the input YAML file
        output_dir (Union[str, Path]): Path to the output directory
        config (dict): Configuration dictionary
        number_of_models (int): Number of models to generate
        seed (int): Seed for the random number generator

    Returns:
        list: The Boltz command
    """

    preprocessing_threads = _normalize_optional_int(
        preprocessing_threads,
        "preprocessing_threads",
    )
    num_workers = _normalize_optional_int(num_workers, "num_workers")
    max_parallel_samples = _normalize_optional_int(
        max_parallel_samples,
        "max_parallel_samples",
    )

    cmd = [
        "boltz",
        "predict",
        str(input_yaml),
        "--out_dir",
        str(output_dir),
        "--override",
        "--write_full_pae",
        "--write_full_pde",
        "--diffusion_samples",
        str(number_of_models),
        "--recycling_steps",
        str(num_recycles),
        # Default 200. Do not lower without validation: 5 sampling steps
        # produced physically invalid structures.
        "--sampling_steps",
        str(int(sampling_steps)),
        "--seed",
        str(seed),
        "--cache",
        _boltz_cache_path(config),
    ]
    if preprocessing_threads is not None:
        cmd += ["--preprocessing-threads", str(preprocessing_threads)]
    if num_workers is not None:
        cmd += ["--num_workers", str(num_workers)]
    if max_parallel_samples is not None:
        cmd += ["--max_parallel_samples", str(max_parallel_samples)]
    if boltz_use_potentials:
        cmd.append("--use_potentials")
    return cmd


def generate_boltz_test_command() -> list:
    """
    Generate the test command for Boltz

    Args:
        None

    Returns:
        list: The Boltz test command
    """

    return [
        "boltz",
        "predict",
        "--help",
    ]

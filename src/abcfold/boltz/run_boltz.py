import json
import logging
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Union

from abcfold.gpus import to_physical
from abcfold.boltz.af3_to_boltz import BoltzYaml
from abcfold.boltz.check_install import ensure_boltz_env
from abcfold.boltz.crystal_modes import apply_crystal_mode

logger = logging.getLogger("logger")


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
        return [None]
    if cuda_devices == "":
        return [""]
    return list(cuda_devices.split(","))


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

        for seed in boltz_yaml.seeds:
            out_file = working_dir.joinpath(f"{input_json.stem}_seed-{seed}.yaml")

            boltz_yaml.write_yaml(out_file)
            logger.info("Running Boltz using seed: %s", seed)
            cmd = (
                generate_boltz_command(
                    out_file,
                    output_dir,
                    config,
                    number_of_models,
                    num_recycles,
                    seed=seed,
                    preprocessing_threads=preprocessing_threads,
                    num_workers=num_workers,
                    max_parallel_samples=max_parallel_samples,
                    sampling_steps=sampling_steps,
                )
                if not test
                else generate_boltz_test_command()
            )

            try:
                if test:
                    logger.info("Skipping Boltz backend execution in test mode")
                    continue
                cuda_devices = normalize_gpus(gpus)
                cmd = _apply_cuda_visible_devices(cmd, cuda_devices)
                stdout = _run_command(env, cmd, output_dir)
                if stdout is None:
                    return False
                if not _has_boltz_structures(output_dir, out_file.stem):
                    logger.error(
                        "Boltz exited without writing a structure for seed %s", seed
                    )
                    return False

            except ValueError as e:
                logger.error("Invalid GPU configuration: %s", e)
                return False
            except subprocess.CalledProcessError:
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
    seed_dirs: dict[int, Path] = {}
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

        inline_threshold = (
            boltz_template_threshold
            if boltz_template_threshold is not None
            else (2.0 if boltz_template_force else None)
        )
        for seed in seeds:
            seed_dir = batch_root / f"seed-{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            boltz_yaml = BoltzYaml(seed_dir, template_threshold=inline_threshold)
            boltz_yaml.json_to_yaml(input_params)
            boltz_yaml.write_yaml(seed_dir / f"{input_json.stem}_seed-{seed}.yaml")
            seed_dirs[int(seed)] = seed_dir

    commands: list[tuple[int, list[str]]] = []
    for seed, seed_dir in sorted(seed_dirs.items()):
        seed_output_dir = batch_root / f"seed-{seed}"
        cmd = (
            generate_boltz_command(
                seed_dir,
                seed_output_dir,
                config,
                number_of_models,
                num_recycles,
                seed=seed,
                preprocessing_threads=preprocessing_threads,
                num_workers=num_workers,
                max_parallel_samples=max_parallel_samples,
                sampling_steps=sampling_steps,
            )
            if not test
            else generate_boltz_test_command()
        )
        commands.append((seed, cmd))

    if test:
        logger.info("Skipping Boltz batch backend execution in test mode")
        return True

    try:
        gpu_slots = _batch_gpu_slots(gpus)
    except ValueError as e:
        logger.error("Invalid GPU configuration: %s", e)
        return False

    def run_seed_command(index: int, seed: int, cmd: list[str]) -> bool:
        logger.info("Running Boltz batch using seed: %s", seed)
        cuda_devices = _seed_gpu(gpu_slots, index)
        run_cmd = _apply_cuda_visible_devices(cmd, cuda_devices)
        stdout = _run_command(env, run_cmd, Path(output_dir))
        return stdout is not None

    try:
        if parallel_seeds and len(gpu_slots) > 1 and len(commands) > 1:
            with ThreadPoolExecutor(
                max_workers=min(len(gpu_slots), len(commands))
            ) as pool:
                for start in range(0, len(commands), len(gpu_slots)):
                    chunk = commands[start:start + len(gpu_slots)]
                    futures = [
                        pool.submit(run_seed_command, index, seed, cmd)
                        for index, (seed, cmd) in enumerate(chunk)
                    ]
                    if not all(future.result() for future in as_completed(futures)):
                        return False
                return True

        for index, (seed, cmd) in enumerate(commands):
            if not run_seed_command(index, seed, cmd):
                return False
    except subprocess.CalledProcessError:
        return False

    logger.info("Boltz batch run complete")
    logger.info("Output files are in %s", output_dir)
    return True


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

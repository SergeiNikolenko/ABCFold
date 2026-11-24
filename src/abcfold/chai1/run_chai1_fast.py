import json
import logging
import subprocess
import tempfile
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Union

from abcfold.chai1.af3_to_chai import ChaiFasta
from abcfold.chai1.check_install import ensure_chai_env

logger = logging.getLogger("logger")


def normalize_device(device: str | None) -> str | None:
    if device is None:
        return None
    if device == "cpu":
        return "cpu"
    if device == "all":
        return "cuda"
    if "," in device:
        device = device.split(",")[0]
    if device.isdigit():
        return f"cuda:{device}"
    return device


def normalize_devices(gpus: str | None) -> list[str | None]:
    if gpus is None:
        return [None]
    if gpus == "cpu":
        return ["cpu"]
    if gpus == "all":
        return ["cuda"]

    devices: list[str | None] = []
    for gpu in gpus.split(","):
        device = normalize_device(gpu.strip())
        if device is not None:
            devices.append(device)
    if not devices:
        raise ValueError("No valid Chai devices were requested.")
    return devices


def run_chai_fast(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    config: dict,
    save_input: bool = False,
    test: bool = False,
    number_of_models: int = 5,
    num_recycles: int = 10,
    use_templates_server: bool = False,
    template_hits_path: Path | None = None,
    device: str | None = None,
    diffusion_steps: int = 200,
    mmseqs_database: Union[str, Path, None] = None,
) -> bool:
    return run_chai_batch(
        [input_json],
        output_dir,
        config=config,
        save_input=save_input,
        test=test,
        number_of_models=number_of_models,
        num_recycles=num_recycles,
        use_templates_server=use_templates_server,
        template_hits_paths=(
            {Path(input_json): template_hits_path}
            if template_hits_path is not None
            else None
        ),
        devices=device,
        nested_outputs=False,
        postprocess=False,
        diffusion_steps=diffusion_steps,
        mmseqs_database=mmseqs_database,
    )


def run_chai_batch(
    input_jsons: Iterable[Union[str, Path]],
    output_dir: Union[str, Path],
    config: dict,
    save_input: bool = False,
    test: bool = False,
    number_of_models: int = 5,
    num_recycles: int = 10,
    use_templates_server: bool = False,
    template_hits_paths: dict[Path, Path | None] | None = None,
    devices: str | None = None,
    nested_outputs: bool = True,
    postprocess: bool = True,
    diffusion_steps: int = 200,
    mmseqs_database: Union[str, Path, None] = None,
) -> bool:
    input_paths = [Path(path) for path in input_jsons]
    if not input_paths:
        logger.error("No Chai input JSON files were provided")
        return False

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_slots = normalize_devices(devices)

    env = None
    kalign_available = True
    if not test:
        logger.debug("Checking if Chai-1 is installed")
        env = ensure_chai_env(config=config)
        kalign_available = env.which("kalign")

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        tasks = _prepare_tasks(
            input_paths=input_paths,
            output_dir=output_dir,
            config=config,
            save_input=save_input,
            use_templates_server=use_templates_server,
            template_hits_paths=template_hits_paths,
            nested_outputs=nested_outputs,
            work_root=temp_dir,
            mmseqs_database=mmseqs_database,
            kalign_available=kalign_available,
            test=test,
        )
        if test:
            logger.info("Skipping Chai fast backend execution in test mode")
            return True

        manifests = _write_device_manifests(
            tasks=tasks,
            devices=device_slots,
            work_root=temp_dir,
            number_of_models=number_of_models,
            num_recycles=num_recycles,
            diffusion_steps=diffusion_steps,
        )
        worker_script = _write_worker_script(temp_dir)
        repo_root = Path(__file__).resolve().parents[2]

        def run_manifest(manifest_path: Path) -> bool:
            assert env is not None
            cmd = [
                "python",
                str(worker_script),
                "--manifest",
                str(manifest_path),
                "--repo-root",
                str(repo_root),
            ]
            extra_env = {}
            downloads = config.get("chai_downloads_dir")
            if downloads:
                extra_env["CHAI_DOWNLOADS_DIR"] = str(Path(str(downloads)).expanduser())
            try:
                env.run(cmd, extra_env=extra_env)
            except subprocess.CalledProcessError as e:
                _write_error_log(Path(output_dir), e)
                return False
            return True

        if len(manifests) > 1:
            with ThreadPoolExecutor(max_workers=len(manifests)) as pool:
                futures = [
                    pool.submit(run_manifest, manifest) for manifest in manifests
                ]
                if not all(future.result() for future in as_completed(futures)):
                    return False
        else:
            if not run_manifest(manifests[0]):
                return False

        missing = [
            task["output_dir"]
            for task in tasks
            if not list(Path(task["output_dir"]).glob("pred.model_idx_*.cif"))
        ]
        if missing:
            logger.error("Chai-1 exited without writing structures in %s", missing)
            return False

        if postprocess:
            _postprocess_cases(tasks, config=config, save_input=save_input)

    logger.info("Chai fast run complete")
    return True


def _prepare_tasks(
    input_paths: list[Path],
    output_dir: Path,
    config: dict,
    save_input: bool,
    use_templates_server: bool,
    template_hits_paths: dict[Path, Path | None] | None,
    nested_outputs: bool,
    work_root: Path,
    mmseqs_database: Union[str, Path, None] = None,
    kalign_available: bool = True,
    test: bool = False,
) -> list[dict]:
    tasks = []
    for input_json in input_paths:
        with input_json.open("r") as f:
            input_params = json.load(f)
        case_id = input_params.get("name") or input_json.stem
        case_output_dir = (
            output_dir / "outputs" / case_id if nested_outputs else output_dir
        )
        case_output_dir.mkdir(parents=True, exist_ok=True)
        working_dir = case_output_dir if save_input else work_root / case_id
        working_dir.mkdir(parents=True, exist_ok=True)

        chai_fasta = ChaiFasta(working_dir, config=config)
        chai_fasta.json_to_fasta(input_json)
        template_hits_path = (
            template_hits_paths.get(input_json) if template_hits_paths else None
        )
        template_cif_store: Path | None = None
        if not test:
            from abcfold.chai1.chai_templates import prepare_chai_templates

            combined_m8, template_cif_store = prepare_chai_templates(
                input_params,
                work_dir=working_dir,
                existing_m8=template_hits_path,
                mmseqs_database=mmseqs_database,
            )
            if combined_m8 is not None:
                template_hits_path = combined_m8

        task_use_templates_server = use_templates_server
        wants_templates = (
            task_use_templates_server or template_hits_path or template_cif_store
        )
        if not kalign_available and wants_templates:
            logger.warning(
                "kalign not found in the Chai-1 environment, skipping template "
                "search. kalign is required for templates with Chai-1."
            )
            task_use_templates_server = False
            template_hits_path = None
            template_cif_store = None

        for seed in chai_fasta.seeds:
            seed_output_dir = case_output_dir / f"chai_output_seed-{seed}"
            seed_output_dir.mkdir(parents=True, exist_ok=True)
            tasks.append(
                {
                    "input_json": str(input_json),
                    "case_id": case_id,
                    "case_output_dir": str(case_output_dir),
                    "fasta": str(chai_fasta.fasta),
                    "msa_dir": str(chai_fasta.working_dir),
                    "constraints": str(chai_fasta.constraints),
                    "output_dir": str(seed_output_dir),
                    "seed": seed,
                    "use_templates_server": task_use_templates_server,
                    "template_hits_path": (
                        str(template_hits_path)
                        if template_hits_path is not None
                        else None
                    ),
                    "template_cif_store": (
                        str(template_cif_store)
                        if template_cif_store is not None
                        else None
                    ),
                }
            )
    return tasks


def _write_device_manifests(
    tasks: list[dict],
    devices: list[str | None],
    work_root: Path,
    number_of_models: int,
    num_recycles: int,
    diffusion_steps: int = 200,
) -> list[Path]:
    per_device: list[list[dict]] = [[] for _ in devices]
    for index, task in enumerate(tasks):
        device_index = index % len(devices)
        task = dict(task)
        task["device"] = devices[device_index]
        task["number_of_models"] = number_of_models
        task["num_recycles"] = num_recycles
        task["diffusion_steps"] = diffusion_steps
        per_device[device_index].append(task)

    manifests = []
    for index, device_tasks in enumerate(per_device):
        if not device_tasks:
            continue
        manifest_path = work_root / f"chai_fast_manifest_{index}.json"
        manifest_path.write_text(json.dumps(device_tasks, indent=2))
        manifests.append(manifest_path)
    return manifests


def _write_worker_script(work_root: Path) -> Path:
    worker_script = work_root / "chai_fast_worker.py"
    worker_script.write_text(textwrap.dedent("""
            import argparse
            import json
            import sys
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--manifest", required=True, type=Path)
            parser.add_argument("--repo-root", required=True, type=Path)
            args = parser.parse_args()

            sys.path.insert(0, str(args.repo_root))

            from abcfold.chai1.chai import run_inference_wrapper

            tasks = json.loads(args.manifest.read_text())
            for task in tasks:
                output_dir = Path(task["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)
                constraint_path = Path(task["constraints"])
                if not constraint_path.exists():
                    constraint_path = None
                template_hits_path = task.get("template_hits_path")
                if template_hits_path is not None:
                    template_hits_path = Path(template_hits_path)
                template_cif_store = task.get("template_cif_store")
                if template_cif_store is not None:
                    template_cif_store = Path(template_cif_store)
                run_inference_wrapper(
                    Path(task["fasta"]),
                    output_dir=output_dir,
                    msa_directory=Path(task["msa_dir"]),
                    constraint_path=constraint_path,
                    use_templates_server=task["use_templates_server"],
                    template_hits_path=template_hits_path,
                    template_cif_store=template_cif_store,
                    num_trunk_recycles=task["num_recycles"],
                    num_diffn_timesteps=task.get("diffusion_steps", 200),
                    num_diffn_samples=task["number_of_models"],
                    seed=task["seed"],
                    device=task["device"],
                )
            """))
    return worker_script


def _postprocess_cases(tasks: list[dict], config: dict, save_input: bool) -> None:
    from abcfold.output.chai import ChaiOutput

    seen_cases: dict[str, dict] = {}
    for task in tasks:
        seen_cases.setdefault(task["case_id"], task)

    for case_id, task in seen_cases.items():
        case_output_dir = Path(task["case_output_dir"])
        chai_output_dirs: list[Union[str, Path]] = list(
            case_output_dir.glob("chai_output*")
        )
        with Path(task["input_json"]).open("r") as f:
            input_params = json.load(f)
        ChaiOutput(chai_output_dirs, input_params, case_id, config, save_input)


def _write_error_log(output_dir: Path, error: subprocess.CalledProcessError) -> None:
    stderr = error.stderr or ""
    output_err_file = output_dir / "chai_fast_error.log"
    output_err_file.write_text(stderr)
    logger.error("Chai fast run failed. Error log is in %s", output_err_file)

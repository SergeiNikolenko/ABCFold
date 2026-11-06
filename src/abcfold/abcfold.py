import json
import logging
import os
import shutil
import socketserver
import sys
import tempfile
import time
import traceback
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from abcfold.argparse_utils import (alphafold_argparse_util,
                                    boltz_argparse_util, chai_argparse_util,
                                    custom_template_argpase_util,
                                    esmfold2_argparse_util,
                                    intellifold_argparse_util,
                                    main_argpase_util, mmseqs2_argparse_util,
                                    msa_argparse_util, openfold_argparse_util,
                                    opendde_argparse_util,
                                    prediction_argparse_util,
                                    protenix_argparse_util,
                                    raise_argument_errors,
                                    rosettafold_argparse_util,
                                    visuals_argparse_util)
from abcfold.output.alphafold3 import AlphafoldOutput
from abcfold.output.boltz import BoltzOutput
from abcfold.output.chai import ChaiOutput
from abcfold.output.esmfold2 import Esmfold2Output
from abcfold.output.intellifold import IntellifoldOutput
from abcfold.output.opendde import OpenDDEOutput
from abcfold.output.openfold3 import OpenfoldOutput
from abcfold.output.protenix import ProtenixOutput
from abcfold.output.rosettafold3 import RosettafoldOutput
from abcfold.report import (EngineRecord, build_report, samples_from_output,
                            write_report)
from abcfold.scripts.abc_script_utils import (check_input_json, make_dir,
                                              make_dummy_af3_db, setup_logger)
from abcfold.settings import Settings, load_settings

logger = setup_logger()

HTML_DIR = Path(__file__).parent / "html"
HTML_TEMPLATE = HTML_DIR.joinpath("abcfold.html.jinja2")
PLOTS_DIR = ".plots"

ENGINE_ORDER = ("alphafold3", "boltz", "chai1", "protenix", "openfold3",
                "rosettafold3", "intellifold", "opendde", "esmfold2")
DISPLAY_NAMES = {
    "alphafold3": "AlphaFold3",
    "boltz": "Boltz",
    "chai1": "Chai-1",
    "protenix": "Protenix",
    "openfold3": "OpenFold3",
    "rosettafold3": "RosettaFold3",
    "intellifold": "IntelliFold",
    "opendde": "OpenDDE",
    "esmfold2": "ESMFold2",
}
MODEL_PREFIX = {
    "alphafold3": "af3",
    "boltz": "boltz",
    "chai1": "chai",
    "protenix": "protenix",
    "openfold3": "openfold",
    "rosettafold3": "rosettafold",
    "intellifold": "intellifold",
    "opendde": "opendde",
    "esmfold2": "esmfold2",
}
EXIT_OK, EXIT_FAILED, EXIT_PARTIAL = 0, 1, 2

ModelOutput = Union[
    AlphafoldOutput,
    BoltzOutput,
    ChaiOutput,
    ProtenixOutput,
    OpenfoldOutput,
    RosettafoldOutput,
    IntellifoldOutput,
    OpenDDEOutput,
    Esmfold2Output,
]


def abcfold_version() -> Optional[str]:
    try:
        from importlib.metadata import version

        return version("abcfold")
    except Exception:  # pragma: no cover - not installed
        return None


# Command line flag -> settings key. CLI values win over config/env values.
CLI_SETTINGS = {
    "model_params": "model_params",
    "database_dir": "database_dir",
    "af3_sif_path": "af3_sif_path",
    "msa_cache_dir": "msa_cache_dir",
    "msa_server_url": "msa_server_url",
    "af3_runner": "af3_runner",
    "af3_image": "af3_image",
    "af3_command": "af3_command",
    "template_mirror_dir": "template_mirror_dir",
    "intellifold_weights": "intellifold_weights",
    "boltz_preprocessing_threads": "boltz_preprocessing_threads",
    "boltz_num_workers": "boltz_num_workers",
    "boltz_max_parallel_samples": "boltz_max_parallel_samples",
    "min_msa_rows": "min_msa_rows",
}


def resolve_settings(args, settings: Optional[Settings] = None) -> Settings:
    """Merge CLI flags into the (read-only) settings and back-fill ``args``."""
    if settings is None:
        settings = load_settings(getattr(args, "config_file", None))
    overrides = {
        key: getattr(args, flag, None)
        for flag, key in CLI_SETTINGS.items()
        if getattr(args, flag, None) not in (None, "", "None")
    }
    settings = settings.apply_overrides(overrides)
    if getattr(args, "protenix_model", None):
        settings.backends["protenix"]["model"] = args.protenix_model
    if getattr(args, "allow_version_mismatch", False):
        settings.values["allow_version_mismatch"] = True
    for flag, key in CLI_SETTINGS.items():
        if getattr(args, flag, None) in (None, "", "None"):
            setattr(args, flag, settings.get(key))
    return settings


@dataclass
class EngineOutcome:
    ok: bool
    output: Optional[ModelOutput] = None
    error: Optional[str] = None


def _error_from_logs(output_dir: Path, since: float = 0.0) -> Optional[str]:
    """Last lines of the newest backend error log written after ``since``."""
    logs = [p for p in output_dir.glob("**/*error*.log") if p.is_file()]
    logs = sorted(
        (p for p in logs if p.stat().st_mtime >= since),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for log in logs:
        try:
            text = log.read_text().strip()
        except OSError:
            continue
        if text:
            return f"{log.name}: {text[-1500:]}"
    return None


def _proteins_have_msas(params: dict) -> bool:
    proteins = [s["protein"] for s in params.get("sequences", []) if "protein" in s]
    return bool(proteins) and all(
        p.get("unpairedMsa") is not None or p.get("unpairedMsaPath") for p in proteins
    )


def _merge_af3_msas(params: dict, data_json: Path) -> dict:
    """Copy MSAs/templates from an AF3 *_data.json into ``params``."""
    from abcfold.msa.resolve import MsaOptions, resolve_msas

    result = resolve_msas(
        params, MsaOptions(source="af3", af3_data_json=str(data_json))
    )
    return result.params


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=4))
    os.replace(tmp, path)


def run(args, settings: Optional[Settings] = None) -> int:
    """Run ABCFold and return the exit code (0 ok, 2 partial, 1 failed)."""
    args.output_dir = Path(args.output_dir)
    settings = resolve_settings(args, settings)
    rt_config = settings.runtime_config()
    args = raise_argument_errors(args)

    if args.mmseqs2:
        logger.info("MMSeqs2 Selected, all other MSAs will be ignored")

    make_dir(args.output_dir, overwrite=args.override)
    make_dir(args.output_dir.joinpath(PLOTS_DIR))

    # abc_<input>.json in the output directory is the single source of truth for
    # every backend; it keeps the MSAs so that a later run can reuse them.
    args.input_json = check_input_json(
        args.input_json,
        output_dir=args.output_dir,
        use_af3_templates=args.use_af3_template_search,
    )
    run_json = Path(args.input_json)
    input_params = json.loads(run_json.read_text())

    name = input_params.get("name")
    if name is None:
        logger.error("Input JSON must contain a 'name' field")
        sys.exit(1)

    msa_section: Dict[str, Any] = {"source": None, "seconds": 0.0, "chains": {}}
    engines = [e for e in ENGINE_ORDER if getattr(args, e, False)]
    records = {e: EngineRecord(engine=e) for e in ENGINE_ORDER}
    outputs: List[Tuple[str, ModelOutput]] = []

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        if args.mmseqs2:
            from abcfold.scripts.add_mmseqs_msa import add_msa_to_json

            input_params = add_msa_to_json(
                input_json=run_json,
                mmseqs_db=args.mmseqs_database,
                templates=args.templates,
                num_templates=args.num_templates,
                chai_template_output=temp_dir.joinpath("all_chains.m8"),
                input_params=input_params,
                output_json=run_json,
                to_file=True,
            )
            msa_section["source"] = "mmseqs2"
        else:
            input_params, msa_section = resolve_input_msas(
                input_params, args, settings
            )
            _write_json(run_json, input_params)

        if args.custom_template is not None:
            from abcfold.scripts.add_custom_template import add_custom_template

            add_custom_template(
                run_json,
                args.target_id,
                args.custom_template,
                args.custom_template_chain,
                output_json=run_json,
                to_file=True,
            )
            input_params = json.loads(run_json.read_text())

        # An MSA that was requested but came back (nearly) empty would let every
        # backend fold single-sequence without noticing.
        from abcfold.msa.guard import (ShallowMsaError, check_input_msa_depth,
                                       min_msa_rows)

        precheck_error = None
        try:
            check_input_msa_depth(input_params, min_msa_rows(rt_config),
                                  context=str(run_json.name))
        except ShallowMsaError as exc:
            logger.error(str(exc))
            precheck_error = str(exc)

        runners: Dict[str, Callable[[], EngineOutcome]] = {
            "alphafold3": lambda: _run_alphafold3(
                args, rt_config, run_json, input_params, temp_dir, name
            ),
            "boltz": lambda: _run_boltz(args, rt_config, run_json, input_params,
                                        name),
            "chai1": lambda: _run_chai(args, rt_config, run_json, input_params,
                                       temp_dir, name),
            "protenix": lambda: _run_protenix(args, rt_config, run_json,
                                              input_params, name),
            "openfold3": lambda: _run_openfold(args, rt_config, run_json,
                                               input_params, temp_dir, name),
            "rosettafold3": lambda: _run_rosettafold(args, rt_config, run_json,
                                                     input_params, name),
            "intellifold": lambda: _run_intellifold(args, rt_config, run_json,
                                                    input_params, name),
            "opendde": lambda: _run_opendde(args, rt_config, run_json,
                                            input_params, name),
            "esmfold2": lambda: _run_esmfold2(args, rt_config, run_json,
                                              input_params, name),
        }

        for engine in engines:
            record = records[engine]
            record.version = settings.backend(engine).get("version")
            logger.info("Running %s", DISPLAY_NAMES[engine])
            start = time.monotonic()
            wall_start = time.time() - 1
            if precheck_error is not None:
                record.status = "failed"
                record.error = precheck_error
                record.seconds = 0.0
                continue
            try:
                outcome = runners[engine]()
            except SystemExit as exc:
                outcome = EngineOutcome(False, error=f"exited with {exc.code}")
            except Exception as exc:  # one backend must not stop the others
                logger.error("%s failed: %s", DISPLAY_NAMES[engine], exc)
                logger.debug(traceback.format_exc())
                outcome = EngineOutcome(False, error=f"{type(exc).__name__}: {exc}")
            record.seconds = time.monotonic() - start
            if outcome.ok:
                record.status = "ok"
                if outcome.output is not None:
                    outputs.append((engine, outcome.output))
                    try:
                        n = sum(
                            1 for models in outcome.output.output.values()
                            for idx in models if not (isinstance(idx, int) and idx < 0)
                        )
                        record.samples = samples_from_output(
                            engine, record.version, outcome.output.output,
                            args.output_dir,
                            seconds_per_sample=(record.seconds / n) if n else None,
                        )
                    except Exception as exc:
                        logger.warning("Could not summarise %s samples: %s",
                                       engine, exc)
            else:
                record.status = "failed"
                record.error = (
                    outcome.error
                    or _error_from_logs(args.output_dir, wall_start)
                    or "backend produced no structures"
                )
                logger.error("%s failed: %s", DISPLAY_NAMES[engine],
                             record.error.splitlines()[0] if record.error else "")
            if engine == "alphafold3" and outcome.ok:
                # AF3 may have produced MSAs that the other backends can reuse
                input_params = json.loads(run_json.read_text())

        report = build_report(
            name=name,
            engines=[records[e] for e in ENGINE_ORDER],
            msa=msa_section,
            input_json=str(run_json),
            output_dir=str(args.output_dir.resolve()),
            abcfold_version=abcfold_version(),
        )
        report_path = write_report(report, args.output_dir / "report.json")
        logger.info("Report written to %s (status: %s)", report_path,
                    report["status"])

        ok = [e for e in engines if records[e].status == "ok"]
        exit_code = (
            EXIT_OK if len(ok) == len(engines)
            else EXIT_PARTIAL if ok else EXIT_FAILED
        )

        if args.output_json:
            shutil.copy(run_json, args.output_json)

        if args.no_visuals:
            logger.info("Visuals disabled")
            return exit_code
        if not outputs:
            logger.error("No models were generated")
            return exit_code
        try:
            render_html(outputs, args.output_dir, no_server=args.no_server)
        except Exception as exc:
            logger.error("Could not build the HTML report: %s", exc)
            logger.debug(traceback.format_exc())
    return exit_code


def resolve_input_msas(input_params: dict, args, settings: Settings):
    """Apply ``--msa`` and return (params, msa report section)."""
    from abcfold.msa.resolve import MsaOptions, parse_chain_paths, resolve_msas
    from abcfold.msa.stats import write_msa_stats

    options = MsaOptions(
        source=getattr(args, "msa", "auto") or "auto",
        cache_dir=settings.get("msa_cache_dir"),
        server_url=settings.get("msa_server_url", "https://api.colabfold.com"),
        user_agent=settings.get("msa_user_agent", "abcfold"),
        paired=not getattr(args, "no_paired_msa", False),
        custom_a3m=parse_chain_paths(getattr(args, "msa_a3m", None)),
        custom_paired_a3m=parse_chain_paths(getattr(args, "msa_paired_a3m", None)),
        af3_data_json=getattr(args, "af3_data_json", None),
    )
    result = resolve_msas(input_params, options)
    section = {"source": result.source, "seconds": round(result.seconds, 3),
               "chains": result.chains}
    if result.chains:
        write_msa_stats(args.output_dir / "msa_stats.json", result.chains,
                        result.source, result.seconds)
    return result.params, section


def _run_alphafold3(args, rt_config, run_json: Path, input_params: dict,
                    temp_dir: Path, name: str) -> EngineOutcome:
    runner = rt_config.get("af3_runner") or "docker"
    sif = args.af3_sif_path
    if runner == "alphafast" or (sif not in (None, "", "None") and runner == "docker"):
        return _run_alphafold3_legacy(args, rt_config, run_json, input_params,
                                      temp_dir, name)

    from abcfold.alphafold3.af3_adapter import Af3Config, run_af3

    image = rt_config.get("af3_image")
    if runner in ("apptainer", "singularity") and sif not in (None, "", "None"):
        image = sif
    cfg = Af3Config.from_runtime_config(
        rt_config,
        runner=runner,
        image=image,
        gpus=args.gpus,
        num_diffusion_samples=args.number_of_models,
        num_recycles=args.num_recycles,
        save_distogram=args.save_distogram,
    )
    if args.af3_data_pipeline_only:
        mode = "data_pipeline"
    elif args.af3_inference_only or (
        _proteins_have_msas(input_params) and not cfg.db_dir
    ):
        mode = "inference"
    else:
        mode = "full"
    result = run_af3(input_params, args.output_dir / "af3_run", cfg, mode=mode)

    if result.data_json is not None and not _proteins_have_msas(input_params):
        merged = _merge_af3_msas(input_params, result.data_json)
        _write_json(run_json, merged)
    if mode == "data_pipeline":
        logger.info("AlphaFold 3 data pipeline written to %s", result.data_json)
        return EngineOutcome(True)
    target = args.output_dir / f"alphafold3_{name}"
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(result.job_dir), target)
    return EngineOutcome(True, AlphafoldOutput(target, input_params, name))


def _run_alphafold3_legacy(args, rt_config, run_json: Path, input_params: dict,
                           temp_dir: Path, name: str) -> EngineOutcome:
    from abcfold.alphafold3.run_alphafold3 import run_alphafold3

    af3_database = args.database_dir
    use_precomputed_msas = args.mmseqs2 or _proteins_have_msas(input_params)
    if use_precomputed_msas:
        af3_database = make_dummy_af3_db(temp_dir)
    af3_output_dir = run_alphafold3(
        input_json=run_json,
        output_dir=args.output_dir,
        model_params=args.model_params,
        database_dir=af3_database,
        number_of_models=args.number_of_models,
        num_recycles=args.num_recycles,
        sif_path=args.af3_sif_path,
        save_distogram=args.save_distogram,
        gpus=args.gpus,
        config=rt_config,
        use_precomputed_msas=use_precomputed_msas,
    )
    if af3_output_dir is None:
        return EngineOutcome(False)
    ao = AlphafoldOutput(af3_output_dir, input_params, name)
    if not use_precomputed_msas:
        _write_json(run_json, _merge_af3_msas(input_params, ao.input_json))
    return EngineOutcome(True, ao)


def _run_boltz(args, rt_config, run_json, input_params, name) -> EngineOutcome:
    from abcfold.boltz.run_boltz import run_boltz

    ok = run_boltz(
        input_json=run_json,
        output_dir=args.output_dir,
        save_input=args.save_input,
        number_of_models=args.number_of_models,
        num_recycles=args.num_recycles,
        gpus=args.gpus,
        config=rt_config,
        boltz_mode=args.boltz_mode,
        boltz_crystal_structure=args.boltz_crystal_structure,
        boltz_ligand_chain=args.boltz_ligand_chain,
        boltz_template_chain_id=args.boltz_template_chain_id,
        boltz_template_id=args.boltz_template_id,
        boltz_template_force=args.boltz_template_force,
        boltz_template_threshold=args.boltz_template_threshold,
        boltz_pocket_radius=args.boltz_pocket_radius,
        boltz_pocket_max_distance=args.boltz_pocket_max_distance,
        boltz_pocket_force=args.boltz_pocket_force,
        preprocessing_threads=args.boltz_preprocessing_threads,
        num_workers=args.boltz_num_workers,
        max_parallel_samples=args.boltz_max_parallel_samples,
        sampling_steps=args.diffusion_steps or 200,
    )
    if not ok:
        return EngineOutcome(False)
    dirs = list(args.output_dir.glob("boltz_results*"))
    return EngineOutcome(True, BoltzOutput(dirs, input_params, name, args.save_input))


def _run_chai(args, rt_config, run_json, input_params, temp_dir, name):
    from abcfold.chai1.run_chai1_fast import run_chai_fast
    from abcfold.output.utils import make_dummy_m8_file

    template_hits_path = None
    if args.templates and args.mmseqs2:
        template_hits_path = temp_dir.joinpath("all_chains.m8")
    elif args.templates:
        template_hits_path = make_dummy_m8_file(run_json, temp_dir)
    ok = run_chai_fast(
        input_json=run_json,
        output_dir=args.output_dir,
        save_input=args.save_input,
        number_of_models=args.number_of_models,
        num_recycles=args.num_recycles,
        template_hits_path=template_hits_path,
        device=args.gpus,
        config=rt_config,
        diffusion_steps=args.diffusion_steps or 200,
    )
    if not ok:
        return EngineOutcome(False)
    dirs = list(args.output_dir.glob("chai_output*"))
    return EngineOutcome(
        True, ChaiOutput(dirs, input_params, name, rt_config, args.save_input)
    )


def _run_protenix(args, rt_config, run_json, input_params, name):
    from abcfold.protenix.run_protenix import run_protenix

    ok = run_protenix(
        input_json=run_json,
        output_dir=args.output_dir,
        save_input=args.save_input,
        number_of_models=args.number_of_models,
        num_recycles=args.num_recycles,
        diffusion_steps=args.diffusion_steps,
        gpus=args.gpus,
        config=rt_config,
    )
    if not ok:
        return EngineOutcome(False)
    dirs = list(args.output_dir.glob("protenix_results*"))
    return EngineOutcome(
        True, ProtenixOutput(dirs, input_params, name, args.save_input)
    )


def _run_openfold(args, rt_config, run_json, input_params, temp_dir, name):
    from abcfold.openfold3.run_openfold3 import run_openfold

    template_hits_path = None
    if args.templates and args.mmseqs2:
        template_hits_path = temp_dir.joinpath("all_chains.m8")
    ok = run_openfold(
        input_json=run_json,
        output_dir=args.output_dir,
        save_input=args.save_input,
        number_of_models=args.number_of_models,
        template_hits_path=template_hits_path,
        input_ckpt=args.inference_ckpt_path,
        num_recycles=args.num_recycles,
        diffusion_steps=args.diffusion_steps,
        fetch_templates=args.openfold_fetch_templates,
        template_mirror_dir=args.template_mirror_dir,
        gpus=args.gpus,
        config=rt_config,
    )
    if not ok:
        return EngineOutcome(False)
    dirs = list(args.output_dir.glob("openfold_results*"))
    return EngineOutcome(
        True, OpenfoldOutput(dirs, input_params, name, args.save_input)
    )


def _run_rosettafold(args, rt_config, run_json, input_params, name):
    from abcfold.rosettafold3.run_rosettafold3 import run_rosettafold

    ok = run_rosettafold(
        input_json=run_json,
        output_dir=args.output_dir,
        save_input=args.save_input,
        number_of_models=args.number_of_models,
        config=rt_config,
    )
    if not ok:
        return EngineOutcome(False)
    dirs = list(args.output_dir.glob("rosettafold_results*"))
    return EngineOutcome(
        True, RosettafoldOutput(dirs, input_params, name, args.save_input)
    )


def _run_intellifold(args, rt_config, run_json, input_params, name):
    from abcfold.intellifold.run_intellifold import run_intellifold

    ok = run_intellifold(
        input_json=run_json,
        output_dir=args.output_dir,
        save_input=args.save_input,
        number_of_models=args.number_of_models,
        config=rt_config,
    )
    if not ok:
        return EngineOutcome(False)
    dirs = list(args.output_dir.glob("intellifold_results*"))
    return EngineOutcome(
        True, IntellifoldOutput(dirs, input_params, name, args.save_input)
    )


def _run_opendde(args, rt_config, run_json, input_params, name):
    from abcfold.opendde.run_opendde import run_opendde

    ok = run_opendde(
        input_json=run_json,
        output_dir=args.output_dir,
        save_input=args.save_input,
        number_of_models=args.number_of_models,
        config=rt_config,
    )
    if not ok:
        return EngineOutcome(False)
    dirs = list(args.output_dir.glob("opendde_results*"))
    return EngineOutcome(
        True, OpenDDEOutput(dirs, input_params, name, args.save_input)
    )


def _run_esmfold2(args, rt_config, run_json, input_params, name):
    from abcfold.esmfold2.run_esmfold2 import run_esmfold2

    ok = run_esmfold2(
        input_json=run_json,
        output_dir=args.output_dir,
        save_input=args.save_input,
        number_of_models=args.number_of_models,
        config=rt_config,
    )
    if not ok:
        return EngineOutcome(False)
    dirs = list(args.output_dir.glob("esmfold2_results*"))
    return EngineOutcome(
        True, Esmfold2Output(dirs, input_params, name, args.save_input)
    )


def render_html(outputs: List[Tuple[str, ModelOutput]], output_dir: Path,
                no_server: bool = False) -> None:
    """Build the interactive HTML page (and optionally serve it)."""
    from abcfold.html.html_utils import (PORT, NoCacheHTTPRequestHandler,
                                         get_all_cif_files, get_model_data,
                                         get_model_sequence_data,
                                         output_open_html_script, plots,
                                         render_template)
    from abcfold.output.file_handlers import superpose_models
    from abcfold.output.utils import get_gap_indicies, insert_none_by_minus_one

    model_outputs = [out for _, out in outputs]
    plot_dict = plots(model_outputs, output_dir.joinpath(PLOTS_DIR))
    cif_models = [
        cif_file
        for cif_list in get_all_cif_files(model_outputs).values()
        for cif_file in cif_list
    ]
    indicies = get_gap_indicies(*cif_models)
    index_counter = 0

    programs_run: List[str] = []
    combined_models: List[Dict[str, Any]] = []
    for engine, out in outputs:
        programs_run.append(DISPLAY_NAMES[engine])
        for seed in out.output.keys():
            for idx in out.output[seed].keys():
                if isinstance(idx, int) and idx < 0:
                    continue
                files = out.output[seed][idx]
                model = files["cif"]
                model.check_clashes()
                score_file = next(
                    (files[k] for k in ("summary", "json", "scores", "score")
                     if files.get(k) is not None),
                    None,
                )
                plddt = model.residue_plddts
                if len(indicies) > 0:
                    plddt = insert_none_by_minus_one(indicies[index_counter], plddt)
                index_counter += 1
                combined_models.append(get_model_data(
                    model, plot_dict, DISPLAY_NAMES[engine], plddt,
                    files["af3_pae"], score_file, output_dir,
                ))

    prefix_by_display = {DISPLAY_NAMES[e]: MODEL_PREFIX[e] for e in DISPLAY_NAMES}
    models_dir = output_dir.joinpath("output_models")
    models_dir.mkdir(parents=True, exist_ok=True)
    output_models = []
    for model in combined_models:
        cif_file = output_dir.joinpath(model["model_path"])
        out_name = (f"{prefix_by_display[model['model_source']]}_model_"
                    f"{model['model_id'][-1]}.cif")
        shutil.copy(cif_file, models_dir / out_name)
        output_models.append(models_dir / out_name)
    if len(output_models) > 1:
        superpose_models(output_models)

    sequence_data = get_model_sequence_data(cif_models)
    sequence = "".join(sequence_data.values())
    chain_data = {}
    ref = 0
    for key, seq in sequence_data.items():
        chain_data["Chain " + key] = (ref, len(seq) + ref - 1)
        ref += len(seq)
    results_json = json.dumps({
        "sequence": sequence,
        "models": combined_models,
        "plotly_path": Path(plot_dict["plddt"])
        .relative_to(output_dir.resolve())
        .as_posix(),
        "chain_data": chain_data,
    })

    if not output_dir.joinpath(".feature_viewer").exists():
        shutil.copytree(HTML_DIR, output_dir / ".feature_viewer")

    if len(programs_run) > 1:
        programs = ("Structure predictions for: " + ", ".join(programs_run[:-1])
                    + " and " + programs_run[-1])
    else:
        programs = "Structure predictions for: " + programs_run[0]

    html_out = output_dir.joinpath("index.html").resolve()
    render_template(
        HTML_TEMPLATE,
        html_out,
        abcfold_html_dir=".feature_viewer",
        programs=programs,
        results_json=results_json,
        version=abcfold_version() or "2",
    )
    logger.info(f"Output page written to {html_out}")
    output_open_html_script(str(output_dir / "open_output.py"), port=PORT)

    if no_server:
        logger.info("Server disabled")
        logger.info("Run 'python open_output.py' in the output directory to "
                    "view the output pages")
        return

    os.chdir(output_dir)
    try:
        with socketserver.TCPServer(("", PORT), NoCacheHTTPRequestHandler) as httpd:
            logger.info(f"Serving at port {PORT}: http://localhost:{PORT}/index.html")
            logger.info("Press Ctrl+C to stop the server")
            webbrowser.open(f"http://localhost:{PORT}/index.html")
            httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped")


def build_run_parser(parser=None):
    import argparse

    if parser is None:
        parser = argparse.ArgumentParser(
            description=(
                "Run AlphaFold3 / Boltz / Chai-1 / OpenFold3 / Protenix / "
                "RosettaFold3"
            ),
        )
    parser.add_argument(
        "--config-file",
        dest="config_file",
        default=None,
        help=dedent_help(
            "[optional] TOML (or legacy INI) config file. Read only; defaults come "
            "from the packaged data/defaults.toml and backends.toml. Can also be "
            "set with ABCFOLD_CONFIG."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="[optional] Debug logging"
    )
    for util in (
        main_argpase_util,
        msa_argparse_util,
        alphafold_argparse_util,
        boltz_argparse_util,
        chai_argparse_util,
        openfold_argparse_util,
        protenix_argparse_util,
        rosettafold_argparse_util,
        intellifold_argparse_util,
        opendde_argparse_util,
        esmfold2_argparse_util,
        mmseqs2_argparse_util,
        custom_template_argpase_util,
        prediction_argparse_util,
        visuals_argparse_util,
    ):
        parser = util(parser)
    return parser


def dedent_help(text: str) -> str:
    return " ".join(text.split())


def main(argv: Optional[List[str]] = None) -> int:
    """Run AlphaFold3 / Boltz / Chai-1 / OpenFold3 / Protenix / RosettaFold3."""
    parser = build_run_parser()
    args = parser.parse_args(argv)
    if getattr(args, "verbose", False):
        logger.setLevel(logging.DEBUG)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())

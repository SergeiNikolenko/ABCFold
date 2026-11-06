import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union

from abcfold.gpus import cuda_env
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

        extra_env = cuda_env(gpus)
        root_dir = config.get("opendde_root_dir")
        if root_dir:
            # Checkpoints and the template assets land here instead of
            # ~/.cache/opendde, so environments and hosts can share them.
            extra_env["OPENDDE_ROOT_DIR"] = str(Path(str(root_dir)).expanduser())

        try:
            env.run(cmd, extra_env=extra_env)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr or ""
            if stderr:
                # The working directory is temporary unless --save_input is
                # set, so the log goes next to the results where it survives.
                output_dir.mkdir(parents=True, exist_ok=True)
                output_err_file = output_dir / "opendde_error.log"
                output_err_file.write_text(stderr)
                logger.error(
                    "OpenDDE run failed. Error log is in %s", output_err_file
                )
            else:
                logger.error("OpenDDE run failed")
            return False

    logger.info("OpenDDE run complete")
    logger.info("Output files are in %s", output_dir)
    return True


def generate_opendde_command(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    config: dict,
    number_of_models: int = 5,
    seeds: Union[list, None] = None,
    use_msa: bool = False,
    num_recycles: int = 10,
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

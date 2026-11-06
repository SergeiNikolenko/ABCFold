import argparse
import logging
import sys
from pathlib import Path
from textwrap import dedent

logger = logging.getLogger("logger")


def validate_json_file(value):
    """
    Validate that the input is a JSON file with a .json suffix.
    """
    if not value.endswith(".json"):
        raise argparse.ArgumentTypeError(
            f"Input file must have a .json suffix: {value}"
        )
    if not Path(value).exists():
        raise argparse.ArgumentTypeError(f"Input file does not exist: {value}")
    return value


def main_argpase_util(parser):
    parser.add_argument(
        "input_json",
        type=validate_json_file,
        help="Path to the input JSON in AlphaFold3 format",
    )
    parser.add_argument("output_dir", help="Path to the output directory")
    parser.add_argument(
        "--override",
        help="[optional] Override the existing output directory, if it exists",
        action="store_true",
    )
    parser.add_argument(
        "--output_json",
        help=dedent("[optional] Specify the path of the output ABCFold json file, this \
        can be used to run subsequent runs of ABCFold with the same input features \
        (e.g. MSA)"),
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="all",
        help="GPU device(s) to use, e.g. 'all', '0', '0,1', 'cpu'",
    )
    parser.add_argument(
        "--allow_version_mismatch",
        action="store_true",
        help=dedent("[optional] Run a backend even if its env has a different \
        version than the pin in backends.toml (a warning is logged)"),
    )

    return parser


def msa_argparse_util(parser):
    parser.add_argument(
        "--msa",
        choices=["auto", "server", "custom", "af3", "none"],
        default="auto",
        help=dedent("[optional] MSA source shared by all backends. 'auto' reuses \
        MSAs present in the input JSON and otherwise leaves MSA generation to each \
        backend; 'server' queries the ColabFold MMseqs2 API (cached); 'custom' \
        reads --msa_a3m files; 'af3' reuses an AlphaFold 3 *_data.json; 'none' \
        runs single-sequence (default: auto)"),
    )
    parser.add_argument(
        "--msa_a3m",
        nargs="+",
        help=dedent("[optional] A3M file(s) for --msa custom, as CHAIN=path \
        (several chains: A,B=path) or a single path for every protein chain"),
    )
    parser.add_argument(
        "--msa_paired_a3m",
        nargs="+",
        help="[optional] Paired A3M file(s) for --msa custom, as CHAIN=path",
    )
    parser.add_argument(
        "--af3_data_json",
        help=dedent("[optional] AlphaFold 3 data-pipeline output (*_data.json or \
        its directory) for --msa af3"),
    )
    parser.add_argument(
        "--msa_cache_dir",
        help="[optional] MSA cache directory (default: ~/.cache/abcfold/msa)",
    )
    parser.add_argument(
        "--msa_server_url",
        help="[optional] ColabFold MMseqs2 API URL (default: https://api.colabfold.com)",
    )
    parser.add_argument(
        "--min_msa_rows",
        type=int,
        default=None,
        help=dedent("[optional] Fail when an MSA was requested but a chain (or a \
        backend's featurised input) has fewer rows; 0 disables (default: 10)"),
    )
    parser.add_argument(
        "--no_paired_msa",
        action="store_true",
        help="[optional] Do not request paired MSAs for heteromeric complexes",
    )
    return parser


def mmseqs2_argparse_util(parser):
    parser.add_argument(
        "--mmseqs2",
        action="store_true",
        help=dedent("[optional] Use MMseqs2 for MSA generation and template \
        searching (if used with --templates flag)"),
    )
    parser.add_argument(
        "--mmseqs_database",
        help=dedent("[optional] The database directory for the generation of the MSA. \
        This is only required if using a local installation of MMseqs2"),
    )
    parser.add_argument(
        "--templates", action="store_true", help="[optional] Enable template search"
    )
    parser.add_argument(
        "--num_templates",
        type=int,
        default=20,
        help="[optional] The number of templates to use (default: 20)",
    )

    return parser


def custom_template_argpase_util(parser):
    parser.add_argument(
        "--target_id",
        nargs="+",
        help=dedent("[conditionally required] The ID of the sequence that the \
        custom template relates to. This is only required if modelling a complex. \
        If providing a list of custom templates, the target_id must be a list of \
        the same length as the custom template list"),
    )
    parser.add_argument(
        "--custom_template",
        nargs="+",
        help=dedent("[optional] Path to a custom template file in mmCif format or a \
        list of paths to custom template files in mmCif format. If providing a list of \
        custom templates, you must also provide a list of custom template chains."),
    )
    parser.add_argument(
        "--custom_template_chain",
        nargs="+",
        help=dedent("[conditionally required] The chain ID of the chain to use in your \
        custom template. This is only required if using a multi-chain template. If \
        providing a list of custom templates, you must also provide a list of custom \
        template chains of the same length as the custom template list"),
    )

    return parser


def prediction_argparse_util(parser):
    parser.add_argument(
        "--number_of_models",
        type=int,
        default=5,
        help=dedent("[optional] The number of models to generate with each method \
        (default: 5)"),
    )
    parser.add_argument(
        "--num_recycles",
        type=int,
        default=10,
        help="[optional] Number of recycles to use during inference (default: 10)",
    )
    parser.add_argument(
        "--diffusion_steps",
        "--sampling_steps",
        dest="diffusion_steps",
        type=int,
        default=None,
        help=dedent("[optional] Diffusion sampling steps for Boltz, Chai-1, \
        Protenix and OpenFold3 (default: each backend's own default, 200)"),
    )
    return parser


def boltz_argparse_util(parser):
    parser.add_argument(
        "-b",
        "--boltz",
        action="store_true",
        help="Run Boltz",
    )
    if "--save_input" not in parser._option_string_actions:
        parser.add_argument(
            "--save_input",
            action="store_true",
            help="Save the input json file",
            default=False,
        )
    parser.add_argument(
        "--boltz_mode",
        choices=[
            "default",
            "1",
            "2",
            "3",
            "template",
            "template_dock",
            "constrained",
            "constrained_dock",
        ],
        default="default",
        help=dedent("[optional] Boltz crystal mode. Use '2'/'template' to add a \
        crystal protein template and optional pocket constraints. Use \
        '3'/'constrained' to force the template and pocket constraints for \
        ligand refinement."),
    )
    parser.add_argument(
        "--boltz_crystal_structure",
        help=dedent("[optional] PDB/mmCIF crystal structure used by --boltz_mode \
        2 or 3 as the Boltz protein template."),
    )
    parser.add_argument(
        "--boltz_ligand_chain",
        help=dedent("[optional] Ligand chain ID in --boltz_crystal_structure. \
        When set, ABCFold derives Boltz pocket constraints from nearby protein \
        residues."),
    )
    parser.add_argument(
        "--boltz_template_chain_id",
        nargs="+",
        help=dedent("[optional] Query protein chain IDs to map to the crystal \
        template. Defaults to Boltz automatic matching."),
    )
    parser.add_argument(
        "--boltz_template_id",
        nargs="+",
        help=dedent("[optional] Template protein chain IDs matching \
        --boltz_template_chain_id. Defaults to Boltz automatic matching."),
    )
    parser.add_argument(
        "--boltz_template_force",
        action="store_true",
        help=dedent("[optional] Force the Boltz template potential in mode 2. \
        Mode 3 enables this automatically."),
    )
    parser.add_argument(
        "--boltz_template_threshold",
        type=float,
        default=2.0,
        help="[optional] Template force threshold in Angstroms (default: 2.0).",
    )
    parser.add_argument(
        "--boltz_pocket_radius",
        type=float,
        default=6.0,
        help=dedent("[optional] Radius in Angstroms used to derive pocket \
        residues around --boltz_ligand_chain (default: 6.0)."),
    )
    parser.add_argument(
        "--boltz_pocket_max_distance",
        type=float,
        default=6.0,
        help=dedent("[optional] Boltz pocket max_distance in Angstroms \
        (default: 6.0)."),
    )
    parser.add_argument(
        "--boltz_pocket_force",
        action="store_true",
        help=dedent("[optional] Force the Boltz pocket potential in mode 2. \
        Mode 3 enables this automatically."),
    )
    parser.add_argument(
        "--boltz_preprocessing_threads",
        type=int,
        default=None,
        help=(
            "[optional] Boltz preprocessing threads for input preparation "
            "(default: from config, 2)."
        ),
    )
    parser.add_argument(
        "--boltz_num_workers",
        type=int,
        default=None,
        help="[optional] Boltz dataloader workers during prediction (default: 2).",
    )
    parser.add_argument(
        "--boltz_max_parallel_samples",
        type=int,
        default=None,
        help=dedent("[optional] Boltz maximum parallel diffusion samples. Leave unset \
        until an OOM boundary has been validated."),
    )

    return parser


def chai_argparse_util(parser):
    parser.add_argument(
        "-c",
        "--chai1",
        action="store_true",
        help="Run Chai-1",
    )
    return parser


def protenix_argparse_util(parser):
    parser.add_argument(
        "-p",
        "--protenix",
        action="store_true",
        help="Run Protenix",
    )
    parser.add_argument(
        "--protenix_model",
        help="[optional] Protenix model name (default: pinned in backends.toml)",
    )
    return parser


def openfold_argparse_util(parser):
    parser.add_argument(
        "-o",
        "--openfold3",
        action="store_true",
        help="Run OpenFold 3",
    )
    parser.add_argument(
        "--inference_ckpt_path",
        help=dedent("Path for model checkpoint to be used for inference. \
    If not specified, will attempt to find or download parameters to \
     ~/.openfold3/"),
    )
    parser.add_argument(
        "--openfold_fetch_templates",
        action="store_true",
        help=dedent("[optional] Allow OpenFold3 to download template structures \
        from RCSB (off by default)"),
    )
    parser.add_argument(
        "--template_mirror_dir",
        help=dedent("[optional] Local mmCIF mirror used for OpenFold3 templates \
        instead of downloading from RCSB"),
    )
    return parser


def rosettafold_argparse_util(parser):
    parser.add_argument(
        "-r",
        "--rosettafold3",
        action="store_true",
        help="Run RosettaFold 3",
    )
    return parser


def intellifold_argparse_util(parser):
    parser.add_argument(
        "-i",
        "--intellifold",
        action="store_true",
        help="Run IntelliFold",
    )
    parser.add_argument(
        "--intellifold_weights",
        help=dedent("[optional] Directory holding the IntelliFold weights. \
        If not specified, parameters are downloaded to ~/.intellifold/model_v2/"),
        default=None,
    )
    return parser


def opendde_argparse_util(parser):
    parser.add_argument(
        "-d",
        "--opendde",
        action="store_true",
        help="Run OpenDDE",
    )
    return parser


def esmfold2_argparse_util(parser):
    parser.add_argument(
        "-e",
        "--esmfold2",
        action="store_true",
        help="Run ESMFold2",
    )
    return parser


def alphafold_argparse_util(parser):
    parser.add_argument(
        "--database",
        help=dedent("[optional] The database directory for the generation of the MSA. \
        This is only required if using the built in AlphaFold3 MSA generation"),
        dest="database_dir",
        default=None,
    )

    parser.add_argument(
        "--model_params",
        help="[required] The directory containing the AlphaFold3 model parameters",
        default=None,
    )

    parser.add_argument(
        "--af3_sif_path",
        help=dedent("[conditionally required] The path to the sif image of AlphaFold3 \
        if using Singularity"),
        default=None,
    )

    parser.add_argument(
        "-a",
        "--alphafold3",
        action="store_true",
        help="Run Alphafold3",
    )

    parser.add_argument(
        "--use_af3_template_search",
        action="store_true",
        help=dedent("If providing your own custom MSA or if you've run `--mmseqs2`, \
        allow Alphafold3 to search for templates"),
    )

    parser.add_argument(
        "--save_distogram",
        action="store_true",
        help="[optional] store AlphaFold3 distograms",
    )

    parser.add_argument(
        "--af3_runner",
        choices=["docker", "apptainer", "singularity", "command", "alphafast"],
        help=dedent("[optional] How to run AlphaFold 3: official image via docker \
        or apptainer/singularity, a site wrapper (--af3_command), or the legacy \
        AlphaFast container (default: from config, docker)"),
    )
    parser.add_argument(
        "--af3_image",
        help="[optional] Docker image name or .sif path for AlphaFold 3",
    )
    parser.add_argument(
        "--af3_command",
        help=dedent("[optional] Wrapper command template for --af3_runner command, \
        e.g. '~/bin/alphafold3 {input_json} {output_dir} {gpus}'. Placeholders: \
        input_json, input_dir, json_name, output_dir, model_dir, db_dir, gpus"),
    )
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument(
        "--af3-data-pipeline-only",
        dest="af3_data_pipeline_only",
        action="store_true",
        help="[optional] Only run the AlphaFold 3 data pipeline (MSA/templates)",
    )
    stage.add_argument(
        "--af3-inference-only",
        dest="af3_inference_only",
        action="store_true",
        help=dedent("[optional] Skip the AlphaFold 3 data pipeline and reuse the \
        MSAs in the input (or from --msa)"),
    )

    return parser


def visuals_argparse_util(parser):
    parser.add_argument(
        "--no_visuals",
        action="store_true",
        help=dedent("[optional] Do not generate the output pages, best for running on \
        a cluster without a display"),
    )

    parser.add_argument(
        "--no_server",
        action="store_true",
        help=dedent("[optional] Do not start a local server to view the results, the \
        output page is still generated and is accessible in the output directory"),
    )
    return parser


def raise_argument_errors(args):
    if (
        not args.alphafold3
        and not args.boltz
        and not args.chai1
        and not args.protenix
        and not args.openfold3
        and not args.rosettafold3
        and not getattr(args, "intellifold", False)
        and not getattr(args, "opendde", False)
        and not getattr(args, "esmfold2", False)
    ):
        logger.info(
            dedent("None of AlphaFold3, Boltz, Chai-1, Protenix, OpenFold3, \
            RosettaFold3, IntelliFold, OpenDDE or ESMFold2 selected. \
            Running AlphaFold3 by default")
        )
        args.alphafold3 = True

    if (
        args.alphafold3
        and not getattr(args, "af3_data_pipeline_only", False)
        and getattr(args, "af3_runner", None) != "command"
        and (not args.model_params or not Path(args.model_params).exists())
        and not args.mmseqs2
    ):
        logger.error(f"Model parameters directory not found: {args.model_params}")
        sys.exit(1)

    if args.templates and not args.mmseqs2 and not args.alphafold3:
        logger.error("Cannot use --templates flag without using MMseqs2 or Alphafold3")
        sys.exit(1)

    if (
        args.templates
        and args.alphafold3
        and not args.mmseqs2
        and not args.use_af3_template_search
    ):
        # Ensure templates are used with Alphafold3 if --templates is set
        args.use_af3_template_search = True

    if args.custom_template_chain and not args.custom_template:
        logger.error("Custom template chain provided without a custom template")
        sys.exit(1)

    if args.use_af3_template_search and not args.alphafold3:
        logger.error(
            "Cannot use the Alphafold3 template search without running Alphafold3"
        )
        sys.exit(1)

    if args.num_templates < 1:
        logger.error("Number of templates must be greater than 0")
        sys.exit(1)

    if args.num_recycles < 1:
        logger.error("Number of recycles must be greater than 0")
        sys.exit(1)

    diffusion_steps = getattr(args, "diffusion_steps", None)
    if diffusion_steps is not None and diffusion_steps < 1:
        logger.error("--diffusion_steps must be greater than 0")
        sys.exit(1)

    if getattr(args, "mmseqs2", False) and getattr(args, "msa", "auto") != "auto":
        logger.error("--mmseqs2 cannot be combined with --msa %s", args.msa)
        sys.exit(1)

    if getattr(args, "msa", "auto") == "custom" and not getattr(args, "msa_a3m", None):
        logger.error("--msa custom requires --msa_a3m")
        sys.exit(1)

    if getattr(args, "msa", "auto") == "af3" and not getattr(args, "af3_data_json", None):
        logger.error("--msa af3 requires --af3_data_json")
        sys.exit(1)

    if args.number_of_models < 1:
        logger.error("Number of models must be greater than 0")
        sys.exit(1)

    if args.boltz_mode != "default" and not args.boltz:
        logger.error("Boltz crystal modes require --boltz")
        sys.exit(1)

    if args.boltz_mode != "default" and not args.boltz_crystal_structure:
        logger.error("--boltz_mode 2/3 requires --boltz_crystal_structure")
        sys.exit(1)

    if args.boltz_mode in {"3", "constrained", "constrained_dock"} and (
        not args.boltz_ligand_chain
    ):
        logger.error("--boltz_mode 3 requires --boltz_ligand_chain")
        sys.exit(1)

    if args.boltz_template_threshold <= 0:
        logger.error("--boltz_template_threshold must be greater than 0")
        sys.exit(1)

    if args.boltz_pocket_radius <= 0:
        logger.error("--boltz_pocket_radius must be greater than 0")
        sys.exit(1)

    if args.boltz_pocket_max_distance <= 0:
        logger.error("--boltz_pocket_max_distance must be greater than 0")
        sys.exit(1)

    for arg_name in [
        "boltz_preprocessing_threads",
        "boltz_num_workers",
        "boltz_max_parallel_samples",
    ]:
        value = getattr(args, arg_name, None)
        if value is None or value == "None":
            setattr(args, arg_name, None)
            continue
        try:
            value = int(value)
        except ValueError:
            logger.error("--%s must be an integer", arg_name)
            sys.exit(1)
        if value < 1:
            logger.error("--%s must be greater than 0", arg_name)
            sys.exit(1)
        setattr(args, arg_name, value)

    return args

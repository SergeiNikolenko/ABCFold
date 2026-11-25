# Run the code for the PAE plots

import csv
import importlib.util
import logging
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Union

from abcfold.output.alphafold3 import AlphafoldOutput
from abcfold.output.boltz import BoltzOutput
from abcfold.output.chai import ChaiOutput
from abcfold.output.cofolding_common import SeedSampleOutput
from abcfold.output.file_handlers import CifFile
from abcfold.output.openfold3 import OpenfoldOutput
from abcfold.output.protenix import ProtenixOutput
from abcfold.output.rosettafold3 import RosettafoldOutput

logger = logging.getLogger(__name__)

CSSPATHS = {
    "A": "paeViewerStandaloneLayoutAF3.css",
    "B": "paeViewerStandaloneLayoutBoltz.css",
    "C": "paeViewerStandaloneLayoutChai.css",
    "O": "paeViewerStandaloneLayoutOpenfold.css",
    "P": "paeViewerStandaloneLayoutProtenix.css",
    "R": "paeViewerStandaloneLayoutRosetta.css",
    "intellifold": "paeViewerStandaloneLayoutIntellifold.css",
    "opendde": "paeViewerStandaloneLayoutOpendde.css",
    "esmfold2": "paeViewerStandaloneLayoutEsmfold2.css",
}

PAEVIEWER = Path(__file__).parent.joinpath(
    "pae-viewer-main", "standalone", "pae_viewer.py"
)

CREATETEMPLATE = Path(__file__).parent.joinpath(
    "pae-viewer-main", "standalone", "create_template.py"
)

INCLUDE_EXTS = [
    ".css",
    ".js",
    ".html",
    ".tpl",
    ".svg",
    ".ico",
    ".png",
]


COLUMNS = [
    "Protein1",
    "SeqPos1",
    "Protein2",
    "SeqPos2",
    "RestraintSatisfied",
]

_LOADED_MODULES: Dict[Path, Any] = {}


def _load_helper_module(path: Path, name: str) -> Any:
    mod = _LOADED_MODULES.get(path)
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load PAE helper module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _LOADED_MODULES[path] = mod
    return mod


def _render_template_in_process(
    title: str,
    standalonecss: str,
    output_file: Path,
    src_path: Path,
) -> None:
    mod = _load_helper_module(CREATETEMPLATE, "_abcfold_pae_create_template")
    args = SimpleNamespace(
        title=title,
        standalonecss=standalonecss,
        output_file=str(output_file.resolve()),
        src_path=str(src_path.resolve()),
    )
    rendered = mod.render_html(args, output_file=str(output_file.resolve()))
    output_file.write_text(rendered)


def _render_pae_viewer_in_process(
    task: Tuple[Path, str, Path, Path, Path, Path]
) -> None:
    cif_path, labels_str, pae_path, output_file, template_file, clashes_csv_file = task
    mod = _load_helper_module(PAEVIEWER, "_abcfold_pae_viewer")
    mod.load_pae_viewer(
        structure_path=cif_path.resolve(),
        chain_labels=labels_str,
        template_file=template_file.resolve(),
        scores_path=pae_path.resolve(),
        crosslinks_path=clashes_csv_file.resolve(),
        output_file=output_file.resolve(),
    )


def _output_template_spec(output: Any) -> Tuple[str, str, str]:
    if isinstance(output, BoltzOutput):
        return "ABCFold - Boltz Output", CSSPATHS["B"], "boltz_template.html"
    if isinstance(output, ChaiOutput):
        return "ABCFold - Chai-1 Output", CSSPATHS["C"], "chai_template.html"
    if isinstance(output, ProtenixOutput):
        return "ABCFold - Protenix Output", CSSPATHS["P"], "protenix_template.html"
    if isinstance(output, OpenfoldOutput):
        return "ABCFold - OpenFold 3 Output", CSSPATHS["O"], "openfold_template.html"
    if isinstance(output, RosettafoldOutput):
        return (
            "ABCFold - RosettaFold 3 Output",
            CSSPATHS["R"],
            "rosettafold_template.html",
        )
    if isinstance(output, AlphafoldOutput):
        return "ABCFold - AlphaFold-3 Output", CSSPATHS["A"], "af3_template.html"
    if isinstance(output, SeedSampleOutput):
        return (
            f"ABCFold - {output.display_name} Output",
            CSSPATHS[output.label],
            f"{output.label}_template.html",
        )
    logger.error("Invalid output type")
    raise ValueError(f"Unsupported output type: {type(output).__name__}")


def create_pae_plots(
    outputs: list,
    output_dir: Union[str, Path],
) -> Dict[str, str]:
    """
    Create PAE html plots for the outputs

    Args:
        outputs: List of outputs to create plots for
        output_dir: Output directory for the plots

    Returns:
        Pathway plot dictionary with the key as the plot path and value as the plot path

    Raises:
        ValueError: If the output type is invalid
    """
    pathway_plot: Dict[str, str] = {}
    render_tasks: List[Tuple[Path, str, Path, Path, Path, Path]] = []
    template_files: List[Path] = []
    output_dir = Path(output_dir)

    copy_pae_viewer_files(output_dir.joinpath(".pae_viewer"))

    for output in outputs:
        plots_dir = (
            Path(output_dir)
            if output_dir
            else output.output_dir.parent.joinpath(".plots")
        )
        plots_dir.mkdir(exist_ok=True)

        title, css_path, template_name = _output_template_spec(output)
        template_file = plots_dir.joinpath(template_name)
        template_files.append(template_file)
        _render_template_in_process(
            title,
            css_path,
            template_file,
            output_dir.joinpath(".pae_viewer"),
        )

        for seed in output.seeds:
            render_tasks.extend(
                _prepare_render_tasks(
                    output.cif_files[seed],
                    output.af3_pae_files[seed],
                    plots_dir,
                    pathway_plot,
                    template_file,
                    is_af3=True,
                )
            )

    if len(render_tasks) > 1:
        with ThreadPoolExecutor(max_workers=min(8, len(render_tasks))) as pool:
            list(pool.map(_render_pae_viewer_in_process, render_tasks))
    else:
        for task in render_tasks:
            _render_pae_viewer_in_process(task)

    for template_file in template_files:
        template_file.unlink(missing_ok=True)

    for csv_file in output_dir.glob("*.csv"):
        csv_file.unlink(missing_ok=True)

    return pathway_plot


def _prepare_render_tasks(
    cif_files, pae_files, plots_dir, pathway_plot, template_file, is_af3=False
) -> List[Tuple[Path, str, Path, Path, Path, Path]]:
    tasks: List[Tuple[Path, str, Path, Path, Path, Path]] = []
    for cif_file, pae_file in zip(cif_files, pae_files):
        name_stem = (
            f"{pae_file.pathway.stem}"
            f"{'_' + pae_file.pathway.parent.stem if is_af3 else ''}_"
            f"{'af3_' if is_af3 else ''}"
        )
        clashes_csv_file = plots_dir.joinpath(f"{name_stem}clashes.csv")
        clashes_csv(cif_file, clashes_csv_file)

        labels = [f"Chain-{chain}" for chain in cif_file.chain_lengths()]
        plot_pathway = plots_dir.joinpath(f"{name_stem}pae_plot.html")
        pathway_plot[str(cif_file.pathway)] = str(plot_pathway)
        tasks.append(
            (
                cif_file.pathway,
                ";".join(labels),
                pae_file.pathway,
                plot_pathway,
                template_file,
                clashes_csv_file,
            )
        )
    return tasks


def prepare_scripts(
    cif_files, pae_files, plots_dir, pathway_plot, template_file, is_af3=False
):
    scripts = []
    for cif_file, pae_file in zip(cif_files, pae_files):
        name_stem = (
            f"{pae_file.pathway.stem}"
            f"{'_' + pae_file.pathway.parent.stem if is_af3 else ''}_"
            f"{'af3_' if is_af3 else ''}"
        )

        clashes_csv_file = plots_dir.joinpath(f"{name_stem}clashes.csv")
        clashes_csv(cif_file, clashes_csv_file)

        labels = [f"Chain-{chain}" for chain in cif_file.chain_lengths()]
        plot_pathway = plots_dir.joinpath(f"{name_stem}pae_plot.html")
        pae_viewer_script = get_pae_run_script(
            cif_file.pathway,
            labels,
            pae_file.pathway,
            plot_pathway,
            template_file,
            clashes_csv_file,
        )
        pathway_plot[str(cif_file.pathway)] = str(plot_pathway)
        scripts.append(pae_viewer_script)
    return scripts


def run_script(cmd):
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        stdout, stderr = proc.communicate()
        proc.wait()

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd, stderr)


def get_pae_run_script(
    cif_path: Union[str, Path],
    labels: List[str],
    pae_path: Union[str, Path],
    output_file: Union[str, Path],
    template_file: Union[str, Path],
    clashes_csv_file: Union[str, Path],
):
    cif_path = Path(cif_path)
    pae_path = Path(pae_path)
    output_file = Path(output_file)
    template_file = Path(template_file)
    clashes_csv_file = Path(clashes_csv_file)

    return [
        sys.executable,
        str(PAEVIEWER.resolve()),
        "--structure",
        str(cif_path.resolve()),
        "--labels",
        ";".join(labels),
        "--scores",
        str(pae_path.resolve()),
        "--output_file",
        str(output_file.resolve()),
        "--template_file",
        str(template_file.resolve()),
        "--crosslinks",
        str(clashes_csv_file.resolve()),
    ]


def get_template_run_script(
    title: str,
    standalonecss: str,
    output_file: Union[str, Path],
    src_path: Union[str, Path],
):
    output_file = Path(output_file)
    src_path = Path(src_path)
    return [
        sys.executable,
        str(CREATETEMPLATE.resolve()),
        "--title",
        title,
        "--standalonecss",
        standalonecss,
        "--output_file",
        str(output_file.resolve()),
        "--src_path",
        str(src_path.resolve()),
    ]


def make_dir(output_dir: Union[str, Path]):
    output_dir = Path(output_dir)
    if output_dir.exists():
        return output_dir
    output_dir.mkdir(parents=True)
    return output_dir


def copy_pae_viewer_files(output_dir: Union[str, Path]):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    pae_viewer_main_pathway = Path(__file__).parent.joinpath("pae-viewer-main")
    for file_ in pae_viewer_main_pathway.rglob("*"):
        if file_.suffix in INCLUDE_EXTS:
            rel = file_.relative_to(pae_viewer_main_pathway)
            new_file = output_dir.joinpath(rel)
            if not new_file.exists():
                new_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_, new_file)


def create_subdirs(output_dir: Union[str, Path], subdirs: List[str]):
    output_dir = Path(output_dir)
    subdirs = subdirs[:-1]
    new_subdirs = output_dir
    for subdir in subdirs:
        new_subdirs = new_subdirs.joinpath(subdir)
        if not new_subdirs.exists():
            new_subdirs.mkdir()


def clashes_csv(cif_file: CifFile, output_name: Union[str, Path]):
    output_name = Path(output_name)
    _, clashes = cif_file.check_clashes()
    with open(output_name, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for atom1, atom2 in clashes:
            fid1 = atom1.get_full_id()
            fid2 = atom2.get_full_id()
            writer.writerow(
                [f"Chain-{fid1[2]}", fid1[3][1], f"Chain-{fid2[2]}", fid2[3][1], False]
            )

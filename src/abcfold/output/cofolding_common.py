import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

from abcfold.output.file_handlers import CifFile, ConfidenceJsonFile
from abcfold.output.utils import Af3Pae

logger = logging.getLogger("logger")

ModelFiles = Dict[str, Union[CifFile, ConfidenceJsonFile]]
# What the results page and the report read: role -> file, as for the other
# backends' outputs.
CollectedFiles = Dict[str, Any]

# Suffix of the AlphaFold3 style PAE written next to each backend confidence
# file; the backend's own file is kept as it is.
AF3_PAE_SUFFIX = "_af3_pae"


def input_dir_for(output_dir: Union[str, Path], label: str) -> Path:
    """Directory a runner writes its converted input to under ``--save_input``.

    Each backend gets its own directory so that collecting one backend's inputs
    never touches ``abc_<input>.json`` or another backend's files.
    """
    return Path(output_dir) / f"{label}_input"


class SeedSampleOutput:
    """
    Shared output handling for backends that write one directory per seed and
    name their files ``<stem><sample_token><n>``.

    Subclasses set ``results_prefix`` to the directory prefix the runner wrote,
    ``label`` to the name used for the collected models, ``display_name`` to
    the name shown in plots and tables, and override ``seed_for`` to read the
    seed from the backend's own directory layout.
    """

    results_prefix: str = ""
    label: str = ""
    display_name: str = ""
    sample_token: str = "_sample-"
    plddt_is_fraction: bool = False

    def __init__(
        self,
        output_dirs: list[Union[str, Path]],
        input_params: dict,
        name: str,
        save_input: bool = False,
    ):
        """
        Args:
            output_dirs (list[Union[str, Path]]): Paths to the run's output
            directories
            input_params (dict): Dictionary containing the input parameters used
            for the run
            name (str): Name given to the run
            save_input (bool): If True, the backend was run with the save_input
            flag

        Attributes:
            output (dict): seed -> model number -> {"cif", "af3_pae", "scores"}
            cif_files (dict): seed -> list of CifFile, in model order
            af3_pae_files (dict): seed -> list of ConfidenceJsonFile, aligned
            with cif_files
            scores_files (dict): seed -> list of ConfidenceJsonFile or None,
            aligned with cif_files
        """
        self.output_dirs = [Path(x) for x in output_dirs]
        self.input_params = input_params
        self.name = name
        self.save_input = save_input

        parent_dir = self.output_dirs[0].parent
        new_parent = parent_dir / f"{self.label}_{self.name}"
        new_parent.mkdir(parents=True, exist_ok=True)

        if self.save_input:
            saved_input = input_dir_for(parent_dir, self.label)
            if saved_input.exists():
                target = new_parent / "input"
                if target.exists():
                    shutil.rmtree(target)
                saved_input.rename(target)

        new_output_dirs = []
        for output_dir in self.output_dirs:
            if output_dir.name.startswith(self.results_prefix):
                new_path = new_parent / output_dir.name
                output_dir.rename(new_path)
                new_output_dirs.append(new_path)
            else:
                new_output_dirs.append(output_dir)
        self.output_dirs = new_output_dirs

        self.output = self.collect(self.process_output())
        self.seeds = list(self.output.keys())

        self.cif_files = {
            seed: [files["cif"] for files in self.output[seed].values()]
            for seed in self.seeds
        }
        self.af3_pae_files = {
            seed: [files["af3_pae"] for files in self.output[seed].values()]
            for seed in self.seeds
        }
        self.scores_files = {
            seed: [files.get("scores") for files in self.output[seed].values()]
            for seed in self.seeds
        }

    def read_cif(self, pathway: Path) -> CifFile:
        if self.plddt_is_fraction:
            return CifFile.from_rosettafold(pathway, self.input_params)
        return CifFile(str(pathway), self.input_params)

    def seed_for(self, output_dir: Path, pathway: Path) -> str:
        """Return the seed a produced file belongs to."""
        return output_dir.name.split("-")[-1].split("_")[-1]

    def sample_number(self, pathway: Path) -> Optional[int]:
        if self.sample_token not in pathway.stem:
            return None
        number_str = pathway.stem.split(self.sample_token)[-1].split("_")[0]
        return int(number_str) if number_str.isdigit() else None

    def pae_data(self, scores: dict):
        """Return the PAE matrix from a backend's confidence file."""
        return scores.get("pae")

    def process_output(self) -> Dict[str, Dict[int, ModelFiles]]:
        """Group the produced files as seed -> model number -> file role."""
        groups: Dict[str, Dict[int, ModelFiles]] = {}
        for pathway in self.output_dirs:
            for output in sorted(pathway.rglob("*")):
                if output.is_dir() or output.stem.endswith(AF3_PAE_SUFFIX):
                    continue
                number = self.sample_number(output)
                if number is None:
                    continue

                files = groups.setdefault(self.seed_for(pathway, output), {})
                model = files.setdefault(number, {})
                if output.suffix == ".cif":
                    if output.stem.endswith("_fixed"):
                        continue
                    cif_file = self.read_cif(output)
                    cif_file.name = (
                        f"{self.label}_{self.seed_for(pathway, output)}_{number}"
                    )
                    model["cif"] = cif_file
                elif output.suffix == ".json":
                    role = "score" if "summary" in output.stem else "pae"
                    model[role] = ConfidenceJsonFile(str(output))

        return groups

    def collect(
        self, groups: Dict[str, Dict[int, ModelFiles]]
    ) -> Dict[str, Dict[int, CollectedFiles]]:
        """Convert each model's PAE to the AlphaFold3 layout.

        Models are keyed by the number the backend gave them, so a missing
        file can never shift one model's confidences onto another. A model
        without a structure or a PAE matrix is left out with a warning, since
        the plots and the results page need both.
        """
        collected: Dict[str, Dict[int, CollectedFiles]] = {}
        for seed in sorted(groups):
            for number in sorted(groups[seed]):
                files = groups[seed][number]
                cif_file = files.get("cif")
                pae_file = files.get("pae")
                if not isinstance(cif_file, CifFile):
                    continue
                matrix = (
                    self.pae_data(pae_file.data)
                    if isinstance(pae_file, ConfidenceJsonFile)
                    else None
                )
                if matrix is None or pae_file is None:
                    logger.warning(
                        "%s seed %s model %s has no PAE matrix; skipping it",
                        self.label, seed, number,
                    )
                    continue
                pae = Af3Pae.from_rosettafold3({"pae": matrix}, cif_file)
                source = Path(pae_file.pathway)
                af3_pae_path = source.with_name(
                    f"{source.stem}{AF3_PAE_SUFFIX}{source.suffix}"
                )
                pae.to_file(af3_pae_path)
                entry: CollectedFiles = {
                    "cif": cif_file,
                    "af3_pae": ConfidenceJsonFile(str(af3_pae_path)),
                }
                if files.get("score") is not None:
                    entry["scores"] = files["score"]
                collected.setdefault(seed, {})[number] = entry
        return collected

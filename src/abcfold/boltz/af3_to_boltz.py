import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

DELIM = "      "

logger = logging.getLogger("logger")

CCD_SMILES_CACHE: dict[str, str | None] = {}


class BoltzYaml:
    """
    Object to convert an AlphaFold3 json file to a boltzmann yaml file.
    """

    def __init__(
        self,
        working_dir: Union[str, Path],
        create_files: bool = True,
        template_threshold: Optional[float] = None,
    ):
        self.working_dir = working_dir
        self.yaml_string: str = ""
        self.msa_file: Optional[Union[str, Path]] = "null"
        self.seeds: list = [42]
        self.__ids: List[Union[str, int]] = []
        self.__id_char: str = "A"
        self.__id_links: Dict[Union[str, int], list] = {}
        self.__create_files = create_files
        self.__non_ligands: List[str] = []
        self.__id_buffer: dict = {}
        self.__template_entries: List[dict] = []
        self.__template_threshold = template_threshold

    @property
    def chain_ids(self) -> List[Union[str, int]]:
        return self.__ids

    @property
    def id_links(self) -> Dict[Union[str, int], list]:
        return self.__id_links

    def ccd_to_smiles(self, ccd_id: str):
        assert isinstance(ccd_id, str), "CCD ID must be a string"
        cache_key = ccd_id.upper()
        if cache_key in CCD_SMILES_CACHE:
            return CCD_SMILES_CACHE[cache_key]
        if not self.__create_files:
            # Post-processing callers (e.g. BoltzOutput) only inspect chain_ids.
            return None

        logger.info("Attempting to retrieve SMILES for CCD %s", ccd_id)
        try:
            response = requests.get(
                f"https://data.rcsb.org/rest/v1/core/chemcomp/{ccd_id}",
                timeout=30,
            )
            if response.status_code == 200:
                descriptor = response.json().get("rcsb_chem_comp_descriptor", {})
                ccd_data = descriptor.get("SMILES") or descriptor.get("SMILES_stereo")
                if ccd_data:
                    logger.info("SMILES retrieved from RCSB: %s", ccd_data)
                    CCD_SMILES_CACHE[cache_key] = ccd_data
                    return ccd_data
        except requests.RequestException as exc:
            logger.warning("RCSB lookup failed for %s: %s", ccd_id, exc)
        CCD_SMILES_CACHE[cache_key] = None
        return None

    def msa_to_file(self, msa: str, file_path: Union[str, Path]):
        """
        Takes a msa string and writes it to a file

        Args:
            msa (str): msa string
            file_path (Union[str, Path]): file path to write the msa to

        Returns:
            None
        """
        out_path = Path(file_path)
        if out_path.exists() and out_path.stat().st_size > 0:
            return
        with open(out_path, "w") as f:
            f.write(msa)

    def _write_template_cifs(self, templates: list) -> List[str]:
        """Write inline mmCIF templates to .cif files for Boltz."""
        cif_paths: List[str] = []
        payload = "\0".join(
            str(t.get("mmcif") or "") for t in templates if isinstance(t, dict)
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
        tmpl_dir = Path(self.working_dir) / f"templates_{digest}"
        if self.__create_files:
            tmpl_dir.mkdir(parents=True, exist_ok=True)
        for idx, template in enumerate(templates):
            mmcif = template.get("mmcif") if isinstance(template, dict) else None
            if not mmcif:
                continue
            cif_file = tmpl_dir / f"template_{idx}.cif"
            if self.__create_files and not cif_file.exists():
                self._normalise_template_cif(mmcif, cif_file)
            cif_paths.append(cif_file.resolve().as_posix())
        return cif_paths

    def _normalise_template_cif(self, mmcif: str, out_file: Path) -> None:
        """Rewrite an inline mmCIF so Boltz gemmi parser can read it."""
        try:
            import gemmi

            doc = gemmi.cif.read_string(mmcif)
            st = gemmi.make_structure_from_block(doc[0])
            if len(st) == 0:
                raise ValueError("template mmCIF has no models")
            st.setup_entities()

            model0 = st[0]
            for entity in st.entities:
                if entity.entity_type.name != "Polymer" or entity.full_sequence:
                    continue
                if not entity.subchains:
                    continue
                first_subchain = entity.subchains[0]
                seq = [
                    residue.name
                    for chain in model0
                    for residue in chain
                    if residue.subchain == first_subchain
                ]
                if seq:
                    entity.full_sequence = seq

            st.make_mmcif_document().write_file(str(out_file))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not normalise template CIF (%s); writing it unchanged", exc
            )
            Path(out_file).write_text(mmcif)

    def add_templates(self) -> str:
        """Build template YAML items for inline protein templates."""
        if not self.__template_entries:
            return ""
        yaml_string = ""
        for entry in self.__template_entries:
            yaml_string += f"{DELIM}- cif: {entry['cif']}\n"
            if self.__template_threshold is not None:
                yaml_string += f"{DELIM}  force: true\n"
                yaml_string += (
                    f"{DELIM}  threshold: {self.__template_threshold}\n"
                )
        return yaml_string

    def json_to_yaml(
        self,
        json_file_or_dict: Union[dict, str, Path],
    ):
        """
        Main function to convert a json file or dict to a yaml string

        Args:
            json_file_or_dict (Union[dict, str, Path]): json file or dict

        Returns:
            str: a string representation of string
        """
        if isinstance(json_file_or_dict, str) or isinstance(json_file_or_dict, Path):
            with open(json_file_or_dict, "r") as f:
                json_dict = json.load(f)
        else:
            json_dict = json_file_or_dict

        self.get_ids(json_dict["sequences"])

        self.yaml_string = ""
        self.__template_entries = []
        constraints_string = ""

        self.yaml_string += self.add_version_number("1")
        for key, value in json_dict.items():
            if key == "modelSeeds":
                if isinstance(value, list):
                    self.seeds = value
                elif isinstance(value, int):
                    self.seeds = [value]
            if key == "sequences":
                if "sequences" not in self.yaml_string:
                    self.yaml_string += self.add_non_indented_string("sequences")
                for sequence_dict in value:
                    if any([key in sequence_dict for key in ["protein", "rna", "dna"]]):
                        self.yaml_string += self.sequence_to_yaml(sequence_dict)
                    if any([key in sequence_dict for key in ["ligand"]]):
                        self.yaml_string += self.add_ligand_information(
                            sequence_dict["ligand"]
                        )
            if key == "bondedAtomPairs" and isinstance(value, list):
                constraints_string += self.bonded_atom_pairs_to_yaml(value)
            if key == "constraints" and isinstance(value, list):
                constraints_string += self.constraints_to_yaml(value)

        if constraints_string:
            self.yaml_string += self.add_non_indented_string("constraints")
            self.yaml_string += constraints_string
        templates_body = ""
        templates = json_dict.get("templates")
        if isinstance(templates, list) and templates:
            templates_body += self.templates_to_yaml(templates)
        templates_body += self.add_templates()
        if templates_body:
            self.yaml_string += self.add_non_indented_string("templates")
            self.yaml_string += templates_body

        return self.yaml_string

    def templates_to_yaml(self, templates: list) -> str:
        yaml_string = ""
        for template in templates:
            if "cif" in template:
                path_key = "cif"
            elif "pdb" in template:
                path_key = "pdb"
            else:
                raise ValueError("Boltz template must include either 'cif' or 'pdb'.")

            yaml_string += f"{DELIM}- {path_key}: {template[path_key]}\n"
            for key in ["chain_id", "template_id", "force", "threshold"]:
                if key in template:
                    yaml_string += self._constraint_key_value(key, template[key])
        return yaml_string

    def constraints_to_yaml(self, constraints: list) -> str:
        yaml_string = ""
        for constraint in constraints:
            if "bond" in constraint:
                yaml_string += self.bond_constraint_to_yaml(constraint["bond"])
            elif "pocket" in constraint:
                yaml_string += self.pocket_constraint_to_yaml(constraint["pocket"])
            elif "contact" in constraint:
                yaml_string += self.contact_constraint_to_yaml(constraint["contact"])
        return yaml_string

    def _yaml_scalar(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return value
        return str(value)

    def _yaml_flow_list(self, values: list) -> str:
        parts = []
        for value in values:
            if isinstance(value, (list, tuple)):
                parts.append(self._yaml_flow_list(list(value)))
            else:
                parts.append(self._yaml_scalar(value))
        return "[" + ", ".join(parts) + "]"

    def _constraint_key_value(self, key: str, value: Any) -> str:
        if isinstance(value, (list, tuple)):
            rendered = self._yaml_flow_list(list(value))
        else:
            rendered = self._yaml_scalar(value)
        return f"{DELIM}{DELIM}{key}: {rendered}\n"

    def bond_constraint_to_yaml(self, bond_dict: dict) -> str:
        yaml_string = self.add_title("bond")
        yaml_string += self._constraint_key_value("atom1", bond_dict["atom1"])
        yaml_string += self._constraint_key_value("atom2", bond_dict["atom2"])
        return yaml_string

    def pocket_constraint_to_yaml(self, pocket_dict: dict) -> str:
        yaml_string = self.add_title("pocket")
        yaml_string += self._constraint_key_value("binder", pocket_dict["binder"])
        yaml_string += self._constraint_key_value("contacts", pocket_dict["contacts"])
        if "max_distance" in pocket_dict:
            yaml_string += self._constraint_key_value(
                "max_distance", pocket_dict["max_distance"]
            )
        if "force" in pocket_dict:
            yaml_string += self._constraint_key_value("force", pocket_dict["force"])
        return yaml_string

    def contact_constraint_to_yaml(self, contact_dict: dict) -> str:
        yaml_string = self.add_title("contact")
        yaml_string += self._constraint_key_value("token1", contact_dict["token1"])
        yaml_string += self._constraint_key_value("token2", contact_dict["token2"])
        if "max_distance" in contact_dict:
            yaml_string += self._constraint_key_value(
                "max_distance", contact_dict["max_distance"]
            )
        if "force" in contact_dict:
            yaml_string += self._constraint_key_value("force", contact_dict["force"])
        return yaml_string

    def bonded_atom_pairs_to_yaml(self, bonded_atom_pairs: list):
        yaml_string = ""
        # counter = 0
        for pair in bonded_atom_pairs:

            if (pair[0][0] == pair[1][0]) and pair[0][1] not in self.__non_ligands:

                if pair[0][0] not in self.__id_links:
                    continue

                # I'm sorry
                if pair[0][0] not in self.__id_buffer:
                    self.__id_buffer[pair[0][0]] = 0
                else:
                    self.__id_buffer[pair[0][0]] += 1

                if self.__id_buffer[pair[0][0]] == 0:
                    first = pair[0][0]
                    second = self.__id_links[pair[0][0]][0]
                else:
                    first, second = (
                        self.__id_links[pair[0][0]][self.__id_buffer[pair[0][0]] - 1],
                        self.__id_links[pair[0][0]][self.__id_buffer[pair[0][0]]],
                    )
                if pair[0][1] < pair[1][1]:
                    pair[0] = [first, 1, pair[0][2]]
                    pair[1] = [second, 1, pair[1][2]]
                else:
                    pair[0] = [first, 1, pair[0][2]]
                    pair[1] = [second, 2, pair[1][2]]
            yaml_string += self.add_title("bond")
            yaml_string += self.add_key_and_value("atom1", pair[0])
            yaml_string += self.add_key_and_value("atom2", pair[1])

        return yaml_string

    def add_version_number(self, version: str):
        """
        Adds the version number to the yaml string

        Args:
            version (str): version number

        Returns:
            str: yaml string
        """
        return f"version: {version}\n"

    def add_non_indented_string(self, string: str):
        """
        Adds the sequence string to the yaml string

        Returns:
            str: yaml string
        """
        return f"{string}:\n"

    def add_id(self, id_: Union[str, list, int]):
        """
        Adds the id to the yaml string

        Args:
            id_ (Union[str, list, int]): id

        Returns:
            str: yaml string

        """

        if isinstance(id_, list):
            self.__ids.extend([id__ for id__ in id_ if id__ not in self.__ids])
            new_id = ", ".join([str(i).replace('"', "").replace("'", "") for i in id_])
        else:
            self.__ids.append(id_) if id_ not in self.__ids else None
            new_id = str(id_).replace('"', "").replace("'", "")

        return (
            f"{DELIM}{DELIM}id: {new_id}\n"
            if not isinstance(id_, list)
            else f"{DELIM}{DELIM}id: [{new_id}]\n"
        )

    def add_sequence(self, sequence: str):
        """
        Adds the sequence to the yaml string

        Args:
            sequence (str): sequence

        Returns:
            str: yaml string

        """
        return f"{DELIM}{DELIM}sequence: {sequence}\n"

    def add_msa(self, msa: Union[str, Path]):
        """
        Adds the msa file_path to the yaml string, double tabbed

        Args:
            msa (str): msa file_path

        Returns:
            str: yaml string
        """
        if not Path(msa).exists() and self.__create_files:
            msg = f"File {msa} does not exist"
            logger.critical(msg)
            raise FileNotFoundError()
        return f"{DELIM}{DELIM}msa: {msa}\n"

    def add_modifications(self, list_of_modifications: list):
        """
        Adds the modifications to the yaml string, double tabbed

        Args:
            list_of_modifications (list): list of modifications

        Returns:
            str: yaml string
        """
        yaml_string = ""
        yaml_string += f"{DELIM}{DELIM}modifications:\n"
        for modification in list_of_modifications:
            if "ptmType" in modification and "ptmPosition" in modification:
                yaml_string += (
                    f"{DELIM}{DELIM}{DELIM}- position: {modification['ptmPosition']}\n"
                    f"{DELIM}{DELIM}{DELIM}  ccd: {modification['ptmType']}\n"
                )
            elif "modificationType" in modification and "basePosition" in modification:
                yaml_string += (
                    f"{DELIM}{DELIM}{DELIM}- position: {modification['basePosition']}\n"
                    f"{DELIM}{DELIM}{DELIM}  ccd: {modification['modificationType']}\n"
                )

        return yaml_string

    def add_key_and_value(self, key: str, value: Any):
        """
        Adds the key and value to the yaml string, double tabbed

        Args:
            key (str): The key on the left of ':'
            value (Any): The value on the right of ':'
        Returns:
            str: yaml string
        """
        if isinstance(value, list):
            parts = [
                f"'{item}'" if isinstance(item, str) else str(item) for item in value
            ]
            value = "[" + ", ".join(parts) + "]"
        if key in {"ccd", "smiles"} and isinstance(value, str):
            val = value.replace('"', '\\"')
            value = f'"{val}"'
        return f"{DELIM}{DELIM}{key}: {value}\n"

    def add_ligand_information(self, ligand_dict: dict, linked_id=None):
        """
        Function to add ligand information to the yaml string

        Args:
            ligand_dict (dict): ligand dict

        Returns:
            str: yaml string
        """

        if "ccdCodes" in ligand_dict and len(ligand_dict["ccdCodes"]) == 0:
            return ""
        yaml_string = ""
        yaml_string += self.add_title("ligand")
        yaml_string += self.add_id(ligand_dict["id"])

        if "smiles" in ligand_dict:
            yaml_string += self.add_key_and_value("smiles", ligand_dict["smiles"])
        elif "ccdCodes" in ligand_dict:
            ccd_codes = (
                [ligand_dict["ccdCodes"]]
                if isinstance(ligand_dict["ccdCodes"], str)
                else ligand_dict["ccdCodes"]
            )
            if linked_id is not None:
                self.__add_linked_ids(linked_id, ligand_dict["id"])
            if linked_id is not None and len(ccd_codes) == 1:
                yaml_string += self.add_key_and_value("ccd", ccd_codes[0])
                return yaml_string
            if len(ccd_codes) == 1:
                smile = self.ccd_to_smiles(ccd_codes[0])
                if smile:
                    yaml_string += self.add_key_and_value("smiles", smile)
                else:
                    yaml_string += self.add_key_and_value("ccd", ccd_codes[0])
            elif isinstance(ligand_dict["ccdCodes"], list):
                yaml_string += self.add_key_and_value("ccd", ligand_dict["ccdCodes"][0])

                yaml_string += self.add_ligand_information(
                    {
                        "id": self.find_next_id(),
                        "ccdCodes": ligand_dict["ccdCodes"][1:],
                    },
                    linked_id=ligand_dict["id"],
                )

        else:

            msg = "Ligand must have either a smiles or ccdCCodes"
            logger.critical(msg)
            raise ValueError()

        return yaml_string

    def add_sequence_information(self, sequence_dict: dict):
        """
        Adds the sequence information of protein, rna, dna to the yaml string

        Args:
            sequence_dict (dict): sequence dict
            msa_file (Union[str, Path]): msa file_path

        Returns:
            str: yaml string
        """
        yaml_string = ""
        yaml_string += self.add_id(sequence_dict["id"])
        yaml_string += (
            self.add_sequence(sequence_dict["sequence"])
            if "sequence" in sequence_dict
            else ""
        )
        if isinstance(sequence_dict["id"], str):
            id_ = [sequence_dict["id"]]
        else:
            id_ = sequence_dict["id"]

        self.__non_ligands.extend(id_)

        if self.msa_file == "empty":
            # Explicit single-sequence mode (MSA source "none")
            yaml_string += f"{DELIM}{DELIM}msa: empty\n"
        elif self.msa_file is not None:
            paired = sequence_dict.get("pairedMsa")
            if paired:
                from abcfold.msa.formats import a3m_to_boltz_csv

                msa_text = a3m_to_boltz_csv(sequence_dict.get("unpairedMsa"), paired)
            else:
                msa_text = sequence_dict["unpairedMsa"]
            (
                self.msa_to_file(msa_text, self.msa_file)
                if self.__create_files
                else None
            )
            yaml_string += self.add_msa(self.msa_file)

        if "modifications" in sequence_dict and sequence_dict["modifications"]:
            yaml_string += self.add_modifications(sequence_dict["modifications"])

        templates = sequence_dict.get("templates")
        if templates:
            for cif_path in self._write_template_cifs(templates):
                self.__template_entries.append(
                    {"cif": cif_path, "chain_id": sequence_dict["id"]}
                )

        return yaml_string

    def add_title(self, name: str):
        """
        Adds the title to the yaml string

        args:
            name (str): name of the title

        Returns:
            str: yaml string
        """
        return f"{DELIM}- {name}:\n"

    def sequence_to_yaml(self, sequence_dict: dict, yaml_string: str = ""):
        """
        Adds the sequence information to the yaml string

        Args:
            sequence_dict (dict): sequence dict
            yaml_string (str): yaml string

        Returns:
            str: yaml string
        """
        for sequence_type, sequence_info_dict in sequence_dict.items():
            yaml_string += self.add_title(sequence_type)
            self.msa_file = None
            if sequence_type == "protein" and (
                "unpairedMsa" in sequence_info_dict or "pairedMsa" in sequence_info_dict
            ):
                unpaired = sequence_info_dict.get("unpairedMsa")
                paired = sequence_info_dict.get("pairedMsa")
                if not unpaired and not paired:
                    self.msa_file = "empty"
                else:
                    suffix = ".csv" if paired else ".a3m"
                    msa_key = f"{unpaired or ''}\0{paired or ''}"
                    digest = hashlib.sha256(msa_key.encode()).hexdigest()[:16]
                    name = f"msa_{digest}"
                    self.msa_file = Path(self.working_dir) / f"{name}{suffix}"

            yaml_string += self.add_sequence_information(sequence_info_dict)

        return yaml_string

    def write_yaml(self, file_path: Union[str, Path]):
        """
        Writes the yaml string to a file

        Args:
            file_path (Union[str, Path]): file path

        Returns:
            None
        """

        assert self.yaml_string, "No yaml string to write to file"
        assert Path(file_path).suffix == ".yaml", "File must have a .yaml extension"
        with open(file_path, "w") as f:
            f.write(self.yaml_string)

    def find_next_id(self):

        if self.__id_char not in self.__ids:
            return self.__id_char
        while self.__id_char in self.__ids:
            self.__id_char = chr(ord(self.__id_char) + 1)
        return self.__id_char

    def get_ids(self, sequences: list):
        for sequence in sequences:
            for key in sequence:
                for key2 in sequence[key]:
                    if key2 == "id":
                        if isinstance(sequence[key][key2], list):
                            self.__ids.extend(sequence[key][key2])
                            continue
                        self.__ids.append(sequence[key][key2])

    def __add_linked_ids(
        self, ligand_id: Union[str, int], linked_ligand_id: Union[str, int]
    ):
        if not self.__id_links:
            self.__id_links[ligand_id] = [linked_ligand_id]
            return
        for id_, value in self.__id_links.items():
            if ligand_id in value:
                self.__id_links[id_].append(linked_ligand_id)

                return

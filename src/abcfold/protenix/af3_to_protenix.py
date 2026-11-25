import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

logger = logging.getLogger("logger")


ION_CCD_CODES = [
    # alkali / alkaline earth
    "LI", "NA", "K", "RB", "CS",
    "MG", "CA", "SR", "BA",
    # transition metals
    "ZN", "FE", "FE2", "FE3", "CU", "MN", "CO", "NI",
    "CD", "HG",
    # halides
    "F", "CL", "BR", "I",
]


class ProtenixJson:
    """
    Object to convert an AlphaFold3 json file to a Protenix JSON file.
    """

    def __init__(self, working_dir: Union[str, Path], create_files: bool = True):
        self.working_dir = working_dir
        self.seeds: list = [42]
        self.__ids: Dict = {}
        self.__id_counter: int = 1
        self.__copies: Dict = {}
        self.__create_files = create_files
        self.protenix_dict: Dict = {}

    @property
    def chain_ids(self) -> Dict:
        return self.__ids

    def _register_entity(self, seq_dict) -> List[str]:
        """Map each chain id to its (entity index, copy index), both 1-based."""
        chain_ids = seq_dict.get("id", [])
        if isinstance(chain_ids, str):
            chain_ids = [chain_ids]
        entity = self.__id_counter
        self.__id_counter += 1
        for copy, chain in enumerate(chain_ids, start=1):
            self.__ids[chain] = entity
            self.__copies[chain] = copy
        return chain_ids

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

    def json_to_json(
        self,
        json_file_or_dict: Union[dict, str, Path],
    ):
        """
        Main function to convert an AF3 json file or dict to a Protenix json string

        Args:
            json_file_or_dict (Union[dict, str, Path]): json file or dict

        Returns:
            Dict: protenix dictionary
        """
        logger.info("Converting input json to a Protenix compatible json file")
        if isinstance(json_file_or_dict, str) or isinstance(json_file_or_dict, Path):
            with open(json_file_or_dict, "r") as f:
                json_dict = json.load(f)
        else:
            json_dict = json_file_or_dict

        protenix_sequences = []
        for key, value in json_dict.items():
            if key == "name":
                self.protenix_dict["name"] = value
            if key == "modelSeeds":
                if isinstance(value, list):
                    self.seeds = value
                elif isinstance(value, int):
                    self.seeds = [value]
                self.protenix_dict["modelSeeds"] = self.seeds

            if key == "sequences":
                for entry in value:
                    if "protein" in entry:
                        protein_entry = self.convert_protein(entry["protein"])
                        protenix_sequences.append(protein_entry)
                    elif "rna" in entry:
                        rna_entry = self.convert_rna(entry["rna"])
                        protenix_sequences.append(rna_entry)
                    elif "dna" in entry:
                        dna_entry = self.convert_dna(entry["dna"])
                        protenix_sequences.append(dna_entry)
                    elif "ligand" in entry:
                        ligand_entry = self.convert_ligand(entry["ligand"])
                        protenix_sequences.append(ligand_entry)

            if key == "bondedAtomPairs" and value is not None:
                contact = self.convert_bonded_atom_pairs(value)
                self.protenix_dict["contact"] = contact

        self.protenix_dict["sequences"] = protenix_sequences

        return self.protenix_dict

    def convert_protein(self, seq_dict) -> Dict[str, Any]:
        sequence = seq_dict["sequence"]
        chain_ids = self._register_entity(seq_dict)
        count = len(chain_ids) if chain_ids else 1

        protein_chain = {
            "sequence": sequence,
            "count": count,
        }

        modifications = seq_dict.get("modifications")
        if modifications:
            prefixed = []
            for mod in modifications:
                m = dict(mod)
                typ = str(m.get("ptmType", ""))
                if not typ.startswith("CCD_"):
                    typ = "CCD_" + typ
                m["ptmType"] = typ
                prefixed.append(m)
            protein_chain["modifications"] = prefixed

        # AF3 inputs may carry the MSA inline or as a file path
        from abcfold.msa.guard import entity_msa_text

        unpaired_msa, paired_msa = entity_msa_text(seq_dict)
        msa_digest = hashlib.sha256(
            f"{unpaired_msa or ''}\0{paired_msa or ''}".encode()
        ).hexdigest()[:16]
        msa_dir = Path(self.working_dir) / f"msa_{msa_digest}"
        if unpaired_msa:
            unpaired_path = msa_dir / "non_pairing.a3m"
            if self.__create_files:
                msa_dir.mkdir(parents=True, exist_ok=True)
                self.msa_to_file(unpaired_msa, unpaired_path)
            protein_chain["unpairedMsaPath"] = unpaired_path.resolve().as_posix()
        if paired_msa:
            paired_path = msa_dir / "pairing.a3m"
            if self.__create_files:
                msa_dir.mkdir(parents=True, exist_ok=True)
                self.msa_to_file(paired_msa, paired_path)
            protein_chain["pairedMsaPath"] = paired_path.resolve().as_posix()

        return {
            "proteinChain": protein_chain
        }

    def convert_rna(self, seq_dict) -> Dict[str, Any]:
        sequence = seq_dict["sequence"]
        chain_ids = self._register_entity(seq_dict)
        count = len(chain_ids) if chain_ids else 1

        rna_chain = {
            "sequence": sequence,
            "count": count,
        }

        modifications = seq_dict.get("modifications")
        if modifications:
            prefixed = []
            for mod in modifications:
                m = dict(mod)
                typ = str(m.get("modificationType", ""))
                if not typ.startswith("CCD_"):
                    typ = "CCD_" + typ
                m["modificationType"] = typ
                prefixed.append(m)
            rna_chain["modifications"] = prefixed

        return {
            "rnaSequence": rna_chain
        }

    def convert_dna(self, seq_dict) -> Dict[str, Any]:
        sequence = seq_dict["sequence"]
        chain_ids = self._register_entity(seq_dict)
        count = len(chain_ids) if chain_ids else 1

        dna_chain = {
            "sequence": sequence,
            "count": count,
        }

        modifications = seq_dict.get("modifications")
        if modifications:
            prefixed = []
            for mod in modifications:
                m = dict(mod)
                typ = str(m.get("modificationType", ""))
                if not typ.startswith("CCD_"):
                    typ = "CCD_" + typ
                m["modificationType"] = typ
                prefixed.append(m)
            dna_chain["modifications"] = prefixed

        return {
            "dnaSequence": dna_chain
        }

    def convert_ligand(self, seq_dict) -> Dict[str, Any]:
        chain_ids = self._register_entity(seq_dict)
        count = len(chain_ids) if chain_ids else 1

        if "ccdCodes" in seq_dict:
            ligand = "_".join(f"CCD_{code}" for code in seq_dict["ccdCodes"])
        else:
            ligand = seq_dict["smiles"]

        # Protenix has no usable "ion" entity: ions are CCD ligands, e.g.
        # {"ligand": {"ligand": "CCD_MG", "count": 1}}
        return {
            "ligand": {
                "ligand": ligand,
                "count": count,
            }
        }

    def convert_bonded_atom_pairs(self,
                                  bonded_atom_pairs: List[List[List[Union[str, int]]]]):
        contacts = []
        for pair in bonded_atom_pairs:
            entity1, position1, atom1 = pair[0]
            entity2, position2, atom2 = pair[1]
            contact = {
                "entity1": self.__ids[entity1],
                "position1": position1,
                "copy1": self.__copies.get(entity1, 1),
                "atom1": atom1,
                "entity2": self.__ids[entity2],
                "position2": position2,
                "copy2": self.__copies.get(entity2, 1),
                "atom2": atom2,
                "max_distance": 6,
                "min_distance": 0
            }
            contacts.append(contact)
        return contacts

    def write_json(self, out_file: Union[str, Path]):
        """
        Write the Protenix json to a file

        Args:
            out_file (Union[str, Path]): output file path

        Returns:
            None
        """

        with open(out_file, "w") as f:
            json.dump([self.protenix_dict], f, indent=4)

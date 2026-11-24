"""Run Boltz-native ligand docking against a crystal receptor template."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.PDBParser import PDBParser

from abcfold.boltz.check_install import ensure_boltz_env


@dataclass(frozen=True)
class ProteinResidue:
    chain_id: str
    pdb_number: int
    sequence_index: int
    residue: Any


@dataclass(frozen=True)
class ProteinChain:
    chain_id: str
    sequence: str
    residues: list[ProteinResidue]


@dataclass(frozen=True)
class DockingInput:
    yaml_path: Path
    command: list[str]
    contacts: list[list[str | int]]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="boltz-dock",
        description=(
            "Dock a ligand SMILES with Boltz while constraining the protein to "
            "a crystal receptor template and the ligand to a pocket."
        ),
    )
    parser.add_argument("receptor", type=Path, help="Crystal receptor PDB file.")
    parser.add_argument("smiles", help="Ligand SMILES to dock into the receptor.")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("boltz_dock"),
        help="Directory for the generated YAML and Boltz outputs.",
    )
    parser.add_argument(
        "--protein_chain",
        action="append",
        dest="protein_chains",
        help=(
            "Protein chain to include. Repeat for multi-chain receptors. "
            "Defaults to all protein chains in the first model."
        ),
    )
    parser.add_argument(
        "--ligand_chain_id",
        default="L",
        help="Boltz chain id assigned to the docked ligand.",
    )
    parser.add_argument(
        "--pocket_residue",
        action="append",
        default=[],
        help=(
            "Pocket residue in CHAIN:RESNUM form. Repeat or use comma-separated "
            "values. RESNUM is PDB numbering by default."
        ),
    )
    parser.add_argument(
        "--pocket_numbering",
        choices=["pdb", "sequence"],
        default="pdb",
        help=(
            "Interpret --pocket_residue numbers as PDB residue numbers or "
            "sequence indices."
        ),
    )
    parser.add_argument(
        "--reference_ligand_chain",
        help="Optional ligand chain in the receptor PDB used to infer pocket residues.",
    )
    parser.add_argument(
        "--pocket_cutoff",
        type=float,
        default=6.0,
        help=(
            "Distance cutoff in Angstrom for --reference_ligand_chain pocket "
            "inference."
        ),
    )
    parser.add_argument(
        "--max_distance",
        type=float,
        default=6.0,
        help="Boltz pocket max_distance in Angstrom.",
    )
    parser.add_argument(
        "--template_threshold",
        type=float,
        default=1.0,
        help="Allowed template deviation in Angstrom when force_template is enabled.",
    )
    parser.add_argument(
        "--no_force_template",
        action="store_true",
        help="Do not force the protein backbone toward the crystal template.",
    )
    parser.add_argument(
        "--no_force_pocket",
        action="store_true",
        help="Do not force the ligand toward the pocket contacts.",
    )
    parser.add_argument(
        "--affinity",
        action="store_true",
        help="Ask Boltz to predict ligand affinity for the generated pose.",
    )
    parser.add_argument(
        "--use_msa_server",
        action="store_true",
        help="Let Boltz query the MSA server instead of using msa: empty.",
    )
    parser.add_argument(
        "--no_use_potentials",
        action="store_true",
        help="Do not pass --use_potentials to boltz predict.",
    )
    parser.add_argument(
        "--diffusion_samples",
        type=int,
        default=25,
        help="Number of Boltz diffusion samples.",
    )
    parser.add_argument(
        "--recycling_steps",
        type=int,
        default=10,
        help="Number of Boltz recycling steps.",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=200,
        help="Number of Boltz diffusion sampling steps.",
    )
    parser.add_argument(
        "--step_scale",
        type=float,
        help="Optional Boltz diffusion step scale.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path.home() / ".boltz",
        help="Boltz cache directory.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices passed to boltz predict.",
    )
    parser.add_argument(
        "--accelerator",
        choices=["gpu", "cpu", "tpu"],
        default="gpu",
        help="Boltz accelerator.",
    )
    parser.add_argument(
        "--output_format",
        choices=["mmcif", "pdb"],
        default="mmcif",
        help="Boltz output structure format.",
    )
    parser.add_argument(
        "--runner",
        choices=["abcfold-env", "path"],
        default="abcfold-env",
        help=(
            "Run through ABCFold's managed Boltz micromamba env, or use boltz "
            "from PATH."
        ),
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Optional read-only ABCFold config (TOML or legacy INI) used when "
        "--runner abcfold-env is selected.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Write the YAML and command file without running Boltz.",
    )
    return parser.parse_args(argv)


def _read_receptor(path: Path) -> Any:
    if path.suffix.lower() != ".pdb":
        raise ValueError("Boltz crystal docking currently expects a receptor PDB file.")
    return PDBParser(QUIET=True).get_structure(path.stem, str(path))


def _one_letter(resname: str) -> str | None:
    return protein_letters_3to1_extended.get(resname.upper())


def _extract_protein_chains(
    structure: Any,
    selected_chain_ids: set[str] | None,
) -> list[ProteinChain]:
    chains = []
    for chain in structure[0]:
        if selected_chain_ids is not None and chain.id not in selected_chain_ids:
            continue

        residues = []
        letters = []
        sequence_index = 1
        for residue in chain:
            if residue.id[0] != " ":
                continue
            letter = _one_letter(residue.resname)
            if letter is None:
                continue
            letters.append(letter)
            residues.append(
                ProteinResidue(
                    chain_id=chain.id,
                    pdb_number=int(residue.id[1]),
                    sequence_index=sequence_index,
                    residue=residue,
                )
            )
            sequence_index += 1

        if residues:
            chains.append(
                ProteinChain(
                    chain_id=chain.id,
                    sequence="".join(letters),
                    residues=residues,
                )
            )

    if not chains:
        raise ValueError("No protein chains were found in the receptor PDB.")
    return chains


def _parse_pocket_residue_tokens(tokens: list[str]) -> list[tuple[str, int]]:
    parsed = []
    for token_group in tokens:
        for token in token_group.split(","):
            token = token.strip()
            if not token:
                continue
            if ":" not in token:
                raise ValueError(f"Pocket residue must use CHAIN:RESNUM: {token}")
            chain_id, residue_number = token.split(":", 1)
            if not chain_id:
                raise ValueError(f"Pocket residue is missing a chain id: {token}")
            parsed.append((chain_id, int(residue_number)))
    return parsed


def _contacts_from_tokens(
    protein_chains: list[ProteinChain],
    tokens: list[str],
    numbering: str,
) -> list[list[str | int]]:
    contacts: list[list[str | int]] = []
    chain_lookup = {chain.chain_id: chain for chain in protein_chains}

    for chain_id, residue_number in _parse_pocket_residue_tokens(tokens):
        chain = chain_lookup.get(chain_id)
        if chain is None:
            raise ValueError(
                f"Pocket chain {chain_id} is not in the receptor proteins."
            )

        for residue in chain.residues:
            number = (
                residue.pdb_number
                if numbering == "pdb"
                else residue.sequence_index
            )
            if number == residue_number:
                contacts.append([chain_id, residue.sequence_index])
                break
        else:
            raise ValueError(
                f"Pocket residue {chain_id}:{residue_number} was not found "
                f"with {numbering} numbering."
            )

    return contacts


def _squared_distance(atom_a: Any, atom_b: Any) -> float:
    delta = atom_a.coord - atom_b.coord
    return float(delta.dot(delta))


def _ligand_atoms_from_chain(structure: Any, chain_id: str) -> list[Any]:
    if chain_id not in structure[0]:
        raise ValueError(f"Reference ligand chain {chain_id} was not found.")
    atoms = [
        atom
        for residue in structure[0][chain_id]
        if residue.id[0] != " "
        for atom in residue
        if atom.element != "H"
    ]
    if not atoms:
        raise ValueError(f"Reference ligand chain {chain_id} has no ligand atoms.")
    return atoms


def _contacts_from_reference_ligand(
    protein_chains: list[ProteinChain],
    structure: Any,
    ligand_chain_id: str,
    cutoff: float,
) -> list[list[str | int]]:
    ligand_atoms = _ligand_atoms_from_chain(structure, ligand_chain_id)
    cutoff_sq = cutoff * cutoff
    contacts: list[list[str | int]] = []

    for chain in protein_chains:
        for protein_residue in chain.residues:
            is_contact = any(
                _squared_distance(atom, ligand_atom) <= cutoff_sq
                for atom in protein_residue.residue
                if atom.element != "H"
                for ligand_atom in ligand_atoms
            )
            if is_contact:
                contacts.append([chain.chain_id, protein_residue.sequence_index])

    if not contacts:
        raise ValueError(
            f"No pocket residues were found within {cutoff:g} A of ligand "
            f"chain {ligand_chain_id}."
        )
    return contacts


def _dedupe_contacts(contacts: list[list[str | int]]) -> list[list[str | int]]:
    seen = set()
    deduped: list[list[str | int]] = []
    for chain_id, residue_index in contacts:
        key = (str(chain_id), int(residue_index))
        if key in seen:
            continue
        seen.add(key)
        deduped.append([key[0], key[1]])
    return deduped


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:g}" if isinstance(value, float) else str(value)
    return json.dumps(str(value))


def _yaml_flow_list(values: list[Any]) -> str:
    rendered = []
    for value in values:
        if isinstance(value, list):
            rendered.append(_yaml_flow_list(value))
        else:
            rendered.append(_yaml_scalar(value))
    return "[" + ", ".join(rendered) + "]"


def _render_yaml(
    receptor: Path,
    protein_chains: list[ProteinChain],
    ligand_chain_id: str,
    smiles: str,
    contacts: list[list[str | int]],
    args: argparse.Namespace,
) -> str:
    lines = ["version: 1", "sequences:"]
    for chain in protein_chains:
        lines.extend([
            "  - protein:",
            f"      id: {_yaml_scalar(chain.chain_id)}",
            f"      sequence: {_yaml_scalar(chain.sequence)}",
        ])
        if not args.use_msa_server:
            lines.append("      msa: empty")

    lines.extend([
        "  - ligand:",
        f"      id: {_yaml_scalar(ligand_chain_id)}",
        f"      smiles: {_yaml_scalar(smiles)}",
        "constraints:",
        "  - pocket:",
        f"      binder: {_yaml_scalar(ligand_chain_id)}",
        f"      contacts: {_yaml_flow_list(contacts)}",
        f"      max_distance: {_yaml_scalar(args.max_distance)}",
        f"      force: {_yaml_scalar(not args.no_force_pocket)}",
        "templates:",
        f"  - pdb: {_yaml_scalar(str(receptor))}",
    ])

    chain_ids = [chain.chain_id for chain in protein_chains]
    template_ids = [f"{chain.chain_id}1" for chain in protein_chains]
    if len(chain_ids) == 1:
        lines.append(f"    chain_id: {_yaml_scalar(chain_ids[0])}")
        lines.append(f"    template_id: {_yaml_scalar(template_ids[0])}")
    else:
        lines.append(f"    chain_id: {_yaml_flow_list(chain_ids)}")
        lines.append(f"    template_id: {_yaml_flow_list(template_ids)}")

    lines.extend([
        f"    force: {_yaml_scalar(not args.no_force_template)}",
        f"    threshold: {_yaml_scalar(args.template_threshold)}",
    ])

    if args.affinity:
        lines.extend([
            "properties:",
            "  - affinity:",
            f"      binder: {_yaml_scalar(ligand_chain_id)}",
        ])

    return "\n".join(lines) + "\n"


def generate_boltz_crystal_dock_command(
    input_yaml: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        "boltz",
        "predict",
        str(input_yaml),
        "--out_dir",
        str(output_dir),
        "--override",
        "--diffusion_samples",
        str(args.diffusion_samples),
        "--recycling_steps",
        str(args.recycling_steps),
        "--sampling_steps",
        str(args.sampling_steps),
        "--cache",
        str(args.cache),
        "--devices",
        str(args.devices),
        "--accelerator",
        args.accelerator,
        "--output_format",
        args.output_format,
    ]
    if args.use_msa_server:
        cmd.append("--use_msa_server")
    if not args.no_use_potentials:
        cmd.append("--use_potentials")
    if args.step_scale is not None:
        cmd.extend(["--step_scale", str(args.step_scale)])
    return cmd


def _load_config(config_file: Path | None) -> dict:
    from abcfold.settings import load_settings

    return load_settings(config_file).runtime_config()


def prepare_crystal_docking_input(args: argparse.Namespace) -> DockingInput:
    receptor = args.receptor.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    structure = _read_receptor(receptor)
    selected_chain_ids = (
        set(args.protein_chains)
        if args.protein_chains is not None
        else None
    )
    protein_chains = _extract_protein_chains(structure, selected_chain_ids)
    contacts = _contacts_from_tokens(
        protein_chains,
        args.pocket_residue,
        args.pocket_numbering,
    )
    if args.reference_ligand_chain is not None:
        contacts.extend(
            _contacts_from_reference_ligand(
                protein_chains,
                structure,
                args.reference_ligand_chain,
                args.pocket_cutoff,
            )
        )
    contacts = _dedupe_contacts(contacts)
    if not contacts:
        raise ValueError(
            "No pocket contacts were provided. Use --pocket_residue or "
            "--reference_ligand_chain."
        )

    yaml_path = out_dir / "boltz_dock.yaml"
    yaml_text = _render_yaml(
        receptor,
        protein_chains,
        args.ligand_chain_id,
        args.smiles,
        contacts,
        args,
    )
    yaml_path.write_text(yaml_text)

    command = generate_boltz_crystal_dock_command(yaml_path, out_dir, args)
    (out_dir / "boltz_dock_command.json").write_text(
        json.dumps(command, indent=2) + "\n"
    )
    return DockingInput(yaml_path=yaml_path, command=command, contacts=contacts)


def run_crystal_docking(args: argparse.Namespace) -> DockingInput:
    docking_input = prepare_crystal_docking_input(args)
    if args.dry_run:
        return docking_input

    if args.runner == "abcfold-env":
        env = ensure_boltz_env(config=_load_config(args.config_file))
        env.run(docking_input.command, capture_output=True)
    else:
        subprocess.run(docking_input.command, check=True)
    return docking_input


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    docking_input = run_crystal_docking(args)
    print(f"Boltz input YAML: {docking_input.yaml_path}")
    print(f"Pocket contacts: {len(docking_input.contacts)}")
    if args.dry_run:
        print("Dry run complete; Boltz was not executed.")


if __name__ == "__main__":
    main()

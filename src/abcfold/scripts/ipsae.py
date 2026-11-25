#!/usr/bin/env python
# mypy: ignore-errors

import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

from abcfold.output.file_handlers import (CifFile, ConfidenceJsonFile,
                                          FileTypes, NpyFile, NpzFile, PklFile,
                                          ResidueCountType)
from abcfold.output.utils import Af3Pae

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger("logger")


class Ipsae():
    def __init__(self,
                 input_model,
                 pae_file,
                 pae_cutoff=5,
                 distance_cutoff=10,
                 pae_format=None):
        self.input_model = input_model
        self.pae_file = pae_file
        self.pae_cutoff = pae_cutoff
        self.distance_cutoff = distance_cutoff

        # Read structure + PAE data
        if isinstance(self.input_model, CifFile):
            # Ipsae only mutates input_params on the wrapper, never the
            # underlying BioPython Structure, so a shallow copy avoids copying
            # thousands of Atom/Residue objects.
            self.struct = copy.copy(self.input_model)
            self.struct.input_params = {}
        else:
            self.struct = CifFile(self.input_model)
        self.process_input()

    @staticmethod
    def ptm_func(x, d0):
        x = np.asarray(x)
        d0 = float(d0)
        return 1.0 / (1.0 + np.square(x / d0))

    @staticmethod
    def calc_d0(L, pair_type='protein'):
        """
        Vectorized d0 calculator.

        - L : scalar or array-like of sequence lengths. Values < 27 are treated as 27.
        - pair_type : string. 'nucleic_acid' (case-insensitive) uses minimum d0 = 2.0;
                    any other value uses minimum d0 = 1.0.

        Returns a NumPy array (or scalar ndarray) of d0 values.
        """
        if not isinstance(pair_type, str):
            raise TypeError("pair_type must be a string")

        L = np.asarray(L, dtype=float)
        L = np.maximum(L, 27.0)

        min_value = 2.0 if pair_type.lower() == 'nucleic_acid' else 1.0
        d0 = 1.24 * np.power(L - 15.0, 1.0 / 3.0) - 1.8
        return np.maximum(min_value, d0)

    @staticmethod
    def nested_chain_map(default, chain_ids):
        return {c1: {c2: default() for c2 in chain_ids if c1 != c2} for c1 in chain_ids}

    @staticmethod
    def pdockq(plddt_mean, n_pairs):
        x = plddt_mean * np.log10(n_pairs)
        return 0.724 / (1 + np.exp(-0.052*(x - 152.611))) + 0.018

    @staticmethod
    def pdockq2(plddt_mean, mean_ptm):
        x2 = plddt_mean * mean_ptm
        return 1.31 / (1 + np.exp(-0.075*(x2 - 84.733))) + 0.005

    @property
    def chain_res_map(self):
        if not hasattr(self, '_chain_res_map'):
            self._chain_res_map = np.asarray([
                chain.id
                for chain in self.struct.get_chains()
                for res in chain if is_aa(res.resname, standard=False)
            ], dtype=object)
        return self._chain_res_map

    @property
    def nres(self):
        if not hasattr(self, '_nres'):
            self._nres = len(self.chain_res_map)
        return self._nres

    @property
    def chain_ids(self):
        if not hasattr(self, '_chain_ids'):
            self._chain_ids = sorted(set(self.chain_res_map))
        return self._chain_ids

    def parse_pae_file(self, pae_path):
        """
        Parse a PAE matrix from one of several supported file formats
        (npz/npy/json/pkl), auto-detecting how the matrix is stored so
        callers don't need to tell us which tool produced it.

        Returns a ``(pae_matrix, af3_data)`` tuple. ``af3_data`` is the
        original dict when the source already carries AlphaFold3-style
        per-token chain/residue ids (e.g. an AlphaFold3 ``*_full_data_*
        .json`` or any file already normalised into that shape) -- this
        lets ``process_input`` reorder tokens to match the structure.
        For every other format, ``af3_data`` is ``None`` and the matrix
        is assumed to already be ordered the same way as the model
        structure.
        """
        suffix = Path(pae_path).suffix[1:]
        if suffix == FileTypes.NPZ.value:
            file_ = NpzFile(pae_path)
        elif suffix == FileTypes.NPY.value:
            file_ = NpyFile(pae_path)
        elif suffix == FileTypes.JSON.value:
            file_ = ConfidenceJsonFile(pae_path)
        elif suffix == FileTypes.PKL.value:
            file_ = PklFile(pae_path)
        else:
            raise ValueError(
                f"Unsupported PAE file format '.{suffix}' for {pae_path}."
            )
        return self._extract_pae_matrix(file_.data, pae_path)

    @staticmethod
    def _extract_pae_matrix(data, source="<data>"):
        """
        Pull a raw PAE matrix out of already-loaded file data, regardless of
        which predictor produced it. See ``parse_pae_file`` for the return
        shape.
        """
        if isinstance(data, list):
            data = data[0]

        if isinstance(data, np.ndarray):
            return np.asarray(data), None

        if isinstance(data, dict):
            if "token_chain_ids" in data and "pae" in data:
                # Already in (or convertible to) AlphaFold3's per-token
                # layout -- keep the full dict so process_input can align
                # chain/residue ordering against the structure.
                return np.asarray(data["pae"]), data
            if "token_pair_pae" in data:
                return np.asarray(data["token_pair_pae"]), None
            if "predicted_aligned_error" in data:
                return np.asarray(data["predicted_aligned_error"]), None
            if "pae" in data:
                return np.asarray(data["pae"]), None

        raise ValueError(f"PAE matrix not found in file: {source}")

    def process_input(self):
        """
        Process input files and setup data types
        """

        self.struct.input_params = {"sequences": []}
        for chain in self.struct.get_chains():
            if all(is_aa(res.resname) for res in chain.get_residues()):
                seq_data = {
                    "protein": {
                        "id": [chain.id],
                        "sequence": "".join(
                            seq1(residue.get_resname()) for residue in chain
                        ),
                    }
                }
            else:
                seq_data = {"ligand": {"id": [chain.id]}}
            self.struct.input_params["sequences"].append(seq_data)

        # Load the PAE matrix, auto-detecting its source format.
        if isinstance(self.pae_file, ConfidenceJsonFile):
            pae_matrix, af3_data = self._extract_pae_matrix(
                self.pae_file.data, self.pae_file
            )
        else:
            pae_matrix, af3_data = self.parse_pae_file(self.pae_file)

        if af3_data is not None:
            self.pae_data = Af3Pae.from_alphafold3(af3_data, self.struct).scores
        else:
            self.pae_data = Af3Pae.from_matrix(pae_matrix, self.struct).scores

        # Extract pLDDTs
        if "plddt" in self.pae_data:
            self.plddt = np.array(self.pae_data["plddt"])
        else:
            plddt = self.struct.get_plddt_per_residue(
                method=ResidueCountType.IPSAE.value
            )
            self.plddt = np.array(
                sum([plddt[chain] for chain in plddt.keys()
                     if not self.struct.check_ligand(chain)
                     ], [])
            )

        # Generate PAE matrix
        if 'pae' in self.pae_data:
            self.pae_matrix = np.array(self.pae_data['pae'])
        elif 'predicted_aligned_error' in self.pae_data:
            self.pae_matrix = np.array(self.pae_data['predicted_aligned_error'])
        else:
            self.pae_matrix = None

        # Some formats include extra (e.g. non-CA/C1) atoms in the matrix.
        # Rather than hard-coding which predictor needs this, mask down to
        # the standard per-residue tokens whenever the matrix is larger
        # than the structure's residue count.
        if self.pae_matrix is not None and self.pae_matrix.shape[0] != self.nres:
            token_mask = self.get_token_mask()
            if token_mask.sum() == self.nres:
                self.pae_matrix = self.pae_matrix[np.ix_(token_mask, token_mask)]
            else:
                raise ValueError(
                    "PAE matrix size "
                    f"({self.pae_matrix.shape[0]}) does not match the number "
                    f"of residues in the structure ({self.nres}), and "
                    "masking to standard residue atoms did not reconcile "
                    f"the difference (mask selects {int(token_mask.sum())})."
                )

        return

    def get_token_mask(self):
        # Mask out non-standard atoms possible
        token_mask = []
        for chain in self.struct.get_chains():
            for residue in chain:
                if is_aa(residue.resname, standard=True):
                    if any((atom.id == "CA") or ("C1" in atom.id) for atom in residue):
                        token_mask.append(1)
                elif is_aa(residue.resname, standard=False):
                    if any((atom.id == "CA") or ("C1" in atom.id) for atom in residue):
                        token_mask.append(1)
                    else:
                        token_mask.append(0)
                elif self.struct.check_ligand(chain):
                    for atom in residue:
                        token_mask.append(0)
                else:
                    for atom in residue:
                        token_mask.append(0)
        return np.array(token_mask, dtype=bool)

    def get_cb_distance(self):
        cb_coords = []
        for chain in self.struct.get_chains():
            for residue in chain:
                if not is_aa(residue.resname, standard=False):
                    continue
                chosen_atom = None
                for atom in residue:
                    if atom.id == "CB":
                        chosen_atom = atom
                        break
                    if "C3" in atom.id:
                        chosen_atom = atom
                        break
                    if residue.resname == "GLY" and atom.id == "CA":
                        chosen_atom = atom
                        break

                if chosen_atom:
                    cb_coords.append(chosen_atom.get_coord())

        if not cb_coords:
            return np.zeros((0, 0), dtype=np.float64)
        coords = np.asarray(cb_coords, dtype=np.float64)
        sq_norms = np.einsum("ij,ij->i", coords, coords)
        sq_dist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (coords @ coords.T)
        np.maximum(sq_dist, 0.0, out=sq_dist)
        distances = np.sqrt(sq_dist, out=sq_dist)
        np.fill_diagonal(distances, 0.0)
        return distances

    def compute_pdockq(self, distances, cutoff=8.0):
        """
        Compute pDockQ and pDockQ2 for all inter-chain interfaces.

        Parameters
        ----------
        distances : (N, N) ndarray
            Matrix of residue–residue Cβ distances.
        cutoff : float
            Distance threshold for residue–residue contacts.

        Returns
        -------
        pdockq_scores, pdockq2_scores : dict[chain1][chain2] → float
        """

        if distances.shape[0] != self.nres or distances.shape[1] != self.nres:
            raise ValueError(
                f"Residue count mismatch: distances.shape={distances.shape}, \
                residues={self.nres}"
            )

        pdockq_scores = self.nested_chain_map(lambda: 0.0, self.chain_ids)
        pdockq2_scores = self.nested_chain_map(lambda: 0.0, self.chain_ids)
        chain_idx = {
            c: np.flatnonzero(self.chain_res_map == c) for c in self.chain_ids
        }
        for c1 in self.chain_ids:
            idx1 = chain_idx[c1]
            if idx1.size == 0:
                continue
            for c2 in self.chain_ids:
                if c1 == c2:
                    continue
                idx2 = chain_idx[c2]
                if idx2.size == 0:
                    continue
                sub_dist = distances[np.ix_(idx1, idx2)]
                close_mask = sub_dist <= cutoff
                n_pairs = int(close_mask.sum())
                if n_pairs == 0:
                    pdockq_scores[c1][c2] = 0.0
                    pdockq2_scores[c1][c2] = 0.0
                    continue
                u1 = idx1[close_mask.any(axis=1)]
                u2 = idx2[close_mask.any(axis=0)]
                uniq_res = np.concatenate((u1, u2))
                mean_plddt = (
                    float(np.mean(self.plddt[uniq_res])) if uniq_res.size > 0 else 0.0
                )

                pdq = self.pdockq(mean_plddt, n_pairs)
                pdockq_scores[c1][c2] = float(np.round(pdq, 4))

                sub_pae = self.pae_matrix[np.ix_(idx1, idx2)][close_mask]
                mean_ptm = float(self.ptm_func(sub_pae, 10.0).sum()) / n_pairs
                pdq2 = self.pdockq2(mean_plddt, mean_ptm)
                pdockq2_scores[c1][c2] = float(np.round(pdq2, 4))
        return pdockq_scores, pdockq2_scores

    def compute_ligand_interface_scores(self):
        """
        Compute ligand interface scores (LIS) between all chain pairs
        using the PAE matrix and a PAE cutoff stored in self.pae_cutoff.
        """

        LIS = self.nested_chain_map(lambda: 0.0, self.chain_ids)

        for chain1 in self.chain_ids:
            for chain2 in self.chain_ids:
                if chain1 == chain2:
                    continue

                # Mask residues from chain1 vs chain2
                mask = (self.chain_res_map[:, None] == chain1) & \
                    (self.chain_res_map[None, :] == chain2)
                selected_pae = self.pae_matrix[mask]

                if selected_pae.size == 0:
                    LIS[chain1][chain2] = 0.0
                    continue

                # Apply cutoff
                valid_pae = selected_pae[selected_pae <= 12]
                if valid_pae.size == 0:
                    LIS[chain1][chain2] = 0.0
                    continue

                # Score = (cutoff - pae) / cutoff
                scores = (12 - valid_pae) / 12
                LIS[chain1][chain2] = float(np.round(np.mean(scores), 4))

        return LIS

    def compute_iptm_ipsae(self, distances):
        """
        Compute iPTM and IPSAE scores for all inter-chain interfaces.

        Parameters
        ----------
        distances : (N, N) ndarray
            Matrix of residue–residue Cβ distances.

        Returns
        -------
        dicts containing per-residue and per-chain-pair scores:
            iptm_d0chn_byres, ipsae_d0chn_byres,
            ipsae_d0dom_byres, ipsae_d0res_byres,
            iptm_d0chn_asym, ipsae_d0chn_asym,
            ipsae_d0dom_asym, ipsae_d0res_asym
        """

        # Initialize dictionaries
        iptm_d0chn_byres = self.nested_chain_map(
            lambda: np.zeros(self.nres), self.chain_ids)
        ipsae_d0chn_byres = self.nested_chain_map(
            lambda: np.zeros(self.nres), self.chain_ids)
        ipsae_d0dom_byres = self.nested_chain_map(
            lambda: np.zeros(self.nres), self.chain_ids)
        ipsae_d0res_byres = self.nested_chain_map(
            lambda: np.zeros(self.nres), self.chain_ids)

        iptm_d0chn_asym = self.nested_chain_map(lambda: 0.0, self.chain_ids)
        ipsae_d0chn_asym = self.nested_chain_map(lambda: 0.0, self.chain_ids)
        ipsae_d0dom_asym = self.nested_chain_map(lambda: 0.0, self.chain_ids)
        ipsae_d0res_asym = self.nested_chain_map(lambda: 0.0, self.chain_ids)

        unique_res_c1 = self.nested_chain_map(lambda: 0, self.chain_ids)
        unique_res_c2 = self.nested_chain_map(lambda: 0, self.chain_ids)
        dist_unique_res_c1 = self.nested_chain_map(lambda: 0, self.chain_ids)
        dist_unique_res_c2 = self.nested_chain_map(lambda: 0, self.chain_ids)

        chain_idx = {
            c: np.flatnonzero(self.chain_res_map == c) for c in self.chain_ids
        }

        # Fully vectorized inter-chain block computation (no Python loops over residues)
        for chain1 in self.chain_ids:
            idx1 = chain_idx[chain1]
            if idx1.size == 0:
                continue
            for chain2 in self.chain_ids:
                if chain1 == chain2:
                    continue
                idx2 = chain_idx[chain2]
                if idx2.size == 0:
                    continue

                sub_pae = self.pae_matrix[np.ix_(idx1, idx2)]
                d0chn = float(self.calc_d0(idx1.size + idx2.size))
                ptm_chn = 1.0 / (1.0 + np.square(sub_pae / d0chn))
                iptm_d0chn_byres[chain1][chain2][idx1] = ptm_chn.mean(axis=1)

                valid_pae = sub_pae < self.pae_cutoff
                counts = valid_pae.sum(axis=1)
                has_valid = counts > 0
                safe_counts = np.where(has_valid, counts, 1)

                ipsae_d0chn_byres[chain1][chain2][idx1] = np.where(
                    has_valid, (ptm_chn * valid_pae).sum(axis=1) / safe_counts, 0.0
                )

                n_u1 = int(has_valid.sum())
                n_u2 = int(valid_pae.any(axis=0).sum())
                unique_res_c1[chain1][chain2] = n_u1
                unique_res_c2[chain1][chain2] = n_u2

                sub_dist = distances[np.ix_(idx1, idx2)]
                valid_dist = valid_pae & (sub_dist < self.distance_cutoff)
                dist_unique_res_c1[chain1][chain2] = int(valid_dist.any(axis=1).sum())
                dist_unique_res_c2[chain1][chain2] = int(valid_dist.any(axis=0).sum())

                d0dom = float(self.calc_d0(n_u1 + n_u2))
                ptm_dom = 1.0 / (1.0 + np.square(sub_pae / d0dom))
                ipsae_d0dom_byres[chain1][chain2][idx1] = np.where(
                    has_valid, (ptm_dom * valid_pae).sum(axis=1) / safe_counts, 0.0
                )

                d0res = self.calc_d0(counts)[:, None]
                ptm_res = 1.0 / (1.0 + np.square(sub_pae / d0res))
                ipsae_d0res_byres[chain1][chain2][idx1] = np.where(
                    has_valid, (ptm_res * valid_pae).sum(axis=1) / safe_counts, 0.0
                )

        # Compute per-chain-pair iPTM / IPSAE as maximum over residues
        for c1 in self.chain_ids:
            for c2 in self.chain_ids:
                if c1 == c2:
                    continue

                iptm_d0chn_asym[c1][c2] = iptm_d0chn_byres[c1][c2].max()
                ipsae_d0chn_asym[c1][c2] = ipsae_d0chn_byres[c1][c2].max()
                ipsae_d0dom_asym[c1][c2] = ipsae_d0dom_byres[c1][c2].max()
                ipsae_d0res_asym[c1][c2] = ipsae_d0res_byres[c1][c2].max()

        return {
            "iptm_d0chn_asym": iptm_d0chn_asym,
            "ipsae_d0chn_asym": ipsae_d0chn_asym,
            "ipsae_d0dom_asym": ipsae_d0dom_asym,
            "ipsae_d0res_asym": ipsae_d0res_asym,
            "unique_residues_chain1": unique_res_c1,
            "unique_residues_chain2": unique_res_c2,
            "dist_unique_residues_chain1": dist_unique_res_c1,
            "dist_unique_residues_chain2": dist_unique_res_c2,
        }

    def main(self, output_csv=None, verbose=True):
        """
        Main function to calculate all the scores output by ipSAE
        """
        distances = self.get_cb_distance()

        pdockq_scores, pdockq2_scores = self.compute_pdockq(distances)
        ligand_interface_scores = self.compute_ligand_interface_scores()
        iptm_ipsae_scores = self.compute_iptm_ipsae(distances)

        self.output_results(pdockq_scores,
                            pdockq2_scores,
                            ligand_interface_scores,
                            iptm_ipsae_scores,
                            verbose,
                            output_csv)

        return {
            "pdockq": pdockq_scores,
            "pdockq2": pdockq2_scores,
            "ligand_interface_scores": ligand_interface_scores,
            "iptm_ipsae_scores": iptm_ipsae_scores
        }

    def output_results(self,
                       pdockq_scores,
                       pdockq2_scores,
                       lis_scores,
                       ipsae_scores,
                       verbose=True,
                       output_csv=None):

        if not verbose and output_csv is None:
            return

        metric_dicts = {}
        if ipsae_scores:
            metric_dicts["ipSAE"] = ipsae_scores['ipsae_d0res_asym']
            metric_dicts["ipSAE_d0chn"] = ipsae_scores['ipsae_d0chn_asym']
            metric_dicts["ipSAE_d0dom"] = ipsae_scores['ipsae_d0dom_asym']
            metric_dicts["iPTM_d0chn"] = ipsae_scores['iptm_d0chn_asym']
        if pdockq_scores:
            metric_dicts["pDockQ"] = pdockq_scores
        if pdockq2_scores:
            metric_dicts["pDockQ2"] = pdockq2_scores
        if lis_scores:
            metric_dicts["LIS"] = lis_scores
        if ipsae_scores:
            metric_dicts['nres1'] = ipsae_scores['unique_residues_chain1']
            metric_dicts['nres2'] = ipsae_scores['unique_residues_chain2']
            metric_dicts['dist1'] = ipsae_scores['dist_unique_residues_chain1']
            metric_dicts['dist2'] = ipsae_scores['dist_unique_residues_chain2']

        rows = []
        chain_pairs = set()
        for metric, scores in metric_dicts.items():
            if scores is None:
                continue
            for ch1, inner in scores.items():
                for ch2 in inner.keys():
                    chain_pairs.add((ch1, ch2))

        # Build rows for each pair
        for ch1, ch2 in sorted(chain_pairs):
            row = {"Chn1": ch1, "Chn2": ch2}

            # Add available metrics
            for metric, scores in metric_dicts.items():
                if scores is None:
                    continue
                try:
                    row[metric] = scores[ch1][ch2]
                except KeyError:
                    row[metric] = None  # missing entry → leave blank

            rows.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(rows)

        if verbose:
            logger.info(df.to_string(index=False))

        if output_csv:
            df.to_csv(output_csv, index=False)

        return df


def main():
    import argparse

    from Bio.PDB import MMCIFIO, PDBParser

    parser = argparse.ArgumentParser(
        prog="ipsae",
        description="Compute IPSAE metrics from structure and PAE file"
    )

    parser.add_argument("input_model", help="Path to input structure (PDB/mmCIF)")
    parser.add_argument(
        "pae_file",
        help="Path to PAE file (npz/npy/json/pkl). The source format "
             "(AlphaFold2/3, Boltz, Chai, ColabFold, OpenFold3, Protenix, "
             "AlphaPulldown, ...) is detected automatically."
    )
    parser.add_argument("--pae_cutoff", type=float, default=10.0,
                        help="PAE cutoff used in some scores (default: 10.0)")
    parser.add_argument("--distance_cutoff", type=float, default=10.0,
                        help="Cβ distance cutoff in Ångström (default: 10.0)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Enable verbose logging/output")
    parser.add_argument("--output", "-o", type=Path,
                        help="Optional output CSV path for results")

    args = parser.parse_args()

    tmp_cif = None
    pdb_path = Path(args.input_model)
    if pdb_path.suffix == ".pdb":
        pdb_parser = PDBParser(QUIET=True)
        structure = pdb_parser.get_structure(pdb_path.stem, str(pdb_path))
        io = MMCIFIO()
        io.set_structure(structure)
        tmp_cif = pdb_path.with_suffix(".cif")
        io.save(str(tmp_cif))
        args.input_model = tmp_cif

    ipsae_calculator = Ipsae(
        args.input_model,
        args.pae_file,
        args.pae_cutoff,
        args.distance_cutoff
    )

    if args.quiet:
        verbose = False
    else:
        verbose = True

    ipsae_calculator.main(verbose=verbose, output_csv=args.output)

    if tmp_cif is not None:
        try:
            Path(tmp_cif).unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()

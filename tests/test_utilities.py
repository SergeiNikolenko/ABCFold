import json
import tempfile
import shutil
from pathlib import Path

from abcfold.output.file_handlers import CifFile, ConfidenceJsonFile
from abcfold.output.utils import Af3Pae
from abcfold.scripts.abc_script_utils import (align_and_map, check_input_json,
                                              extract_sequence_from_mmcif,
                                              get_chains, get_mmcif)


def test_get_chains(test_data):
    chains = get_chains(test_data.test_6BJ9_cif)
    assert all(isinstance(chain, str) for chain in chains)
    assert len(chains) == 2
    assert "A" in chains
    assert "B" in chains


def test_extract_sequence_from_mmcif(test_data):
    with open(test_data.test_6BJ9_cif) as f:
        sequence = extract_sequence_from_mmcif(f)
    assert isinstance(sequence, str)
    assert len(sequence) == 770
    assert sequence.startswith("A")
    assert sequence.endswith("L")
    assert sequence.count("A") == 178
    assert " " not in sequence


def test_align_and_map(test_data):
    with open(test_data.test_6BJ9_cif) as f:
        template_seq = extract_sequence_from_mmcif(f)
    query_seq = template_seq[::-1]
    query_indices, template_indices = align_and_map(query_seq, template_seq)
    assert len(query_indices) == 385
    assert len(template_indices) == 385
    assert all(isinstance(i, int) for i in query_indices)
    assert all(isinstance(i, int) for i in template_indices)
    assert query_indices[0] == 3
    assert template_indices[0] == 2


def test_get_mmcif(test_data):
    cif_str = get_mmcif(test_data.test_6BJ9_cif, "6BJ9", "A", 1, 100)
    assert isinstance(cif_str, str)
    assert len(cif_str) == 110528
    cif_list = cif_str.split("\n")

    assert cif_list[0] == "data_6BJ9"
    assert len(cif_list) == 1479

    cif_str = get_mmcif(test_data.test_6BJ9_metadata_strip_cif, "6BJ9", "A", 1, 100)

    assert "_pdbx_audit_revision_history.revision_date" in cif_str

    cif_str = get_mmcif(test_data.test_1G03_cif, "1G03", "A", 1, 100)
    assert cif_str.startswith("data_1G03")


def test_check_input_json(test_data):
    input_json = test_data.test_inputAmsa_json
    input_json1 = check_input_json(input_json, use_af3_templates=True, test=True)
    input_json2 = check_input_json(input_json, use_af3_templates=False, test=True)

    assert "templates" in input_json1["sequences"][0]["protein"]
    assert input_json1["sequences"][0]["protein"]["templates"] is None

    assert "templates" in input_json2["sequences"][0]["protein"]
    assert input_json2["sequences"][0]["protein"]["templates"] == []


def test_clash_checker(test_data):
    cif_file = Path(test_data.test_boltz_6BJ9_seed_1_).joinpath(
        "predictions", "test_mmseqs", "test_mmseqs_model_0.cif"
    )

    structure = CifFile(cif_file)
    clashes, _ = structure.check_clashes()
    assert isinstance(clashes, list)
    assert len(clashes) == 2


def test_af3_pae_reorder(test_data, tmp_path):
    # reorder_chains() persists to the source file, so work on a copy of the fixture.
    test_cif = tmp_path / "model.cif"
    shutil.copy(
        Path(test_data.test_alphafold3_6BJ9_).joinpath("seed-1_sample-0", "model.cif"),
        test_cif,
    )
    test_pae = Path(test_data.test_alphafold3_6BJ9_).joinpath(
        "seed-1_sample-0", "confidences.json"
    )
    # /home/etk48667/dev/ABCFold/tests/test_data/alphafold3_6BJ9/6bj9_data.json
    input_params_path = Path(test_data.test_alphafold3_6BJ9_).joinpath("6bj9_data.json")

    with open(input_params_path, "r") as f:
        input_params = json.load(f)

    cif = CifFile(test_cif, input_params)
    pae = ConfidenceJsonFile(test_pae)
    pae_to_compare_tci = pae.data["token_chain_ids"][786:]

    cif.reorder_chains(["B", "A", "D", "C"])

    with tempfile.TemporaryDirectory() as temp_dir:
        cif.to_file(Path(temp_dir).joinpath("test.cif"))
        cif_file = Path(temp_dir).joinpath("test.cif")
        structure = CifFile(cif_file, input_params=input_params)
        assert [chain.id for chain in structure.get_chains()] == ["B", "A", "D", "C"]
        structure.update()
        assert [chain.id for chain in structure.get_chains()] == ["B", "A", "D", "C"]
        pae_obj = Af3Pae.from_alphafold3(pae.data, structure)

        assert (
            pae_obj.scores["token_chain_ids"][787 + 50 :]  # noqa: E203
            == pae_to_compare_tci[:51]
        )


def test_limit_msa_depth_and_json():
    from abcfold.scripts.abc_script_utils import (limit_json_msa_depth,
                                                  limit_msa_depth)

    a3m = ">query\nMKVLA\n>h1\nMKVLA\n>h2\nMKVLG\n>h3\nMRVLA\n"
    out = limit_msa_depth(a3m, 2)
    assert out.count(">") == 2
    assert out.startswith(">query")
    assert limit_msa_depth(a3m, None) == a3m
    assert limit_msa_depth(a3m, 0) == a3m

    with tempfile.TemporaryDirectory() as td:
        a3m_path = Path(td) / "m.a3m"
        a3m_path.write_text(a3m)
        inp = Path(td) / "in.json"
        out_p = Path(td) / "out.json"
        inp.write_text(
            json.dumps(
                {
                    "sequences": [
                        {
                            "protein": {
                                "id": "A",
                                "sequence": "MKVLA",
                                "unpairedMsaPath": str(a3m_path),
                                "pairedMsa": a3m,
                            }
                        }
                    ]
                }
            )
        )
        data = limit_json_msa_depth(inp, out_p, 2)
        prot = data["sequences"][0]["protein"]
        assert prot["unpairedMsa"].count(">") == 2
        assert prot["pairedMsa"].count(">") == 2
        assert "unpairedMsaPath" not in prot


def test_get_gap_indicies_empty_and_from_matrix(test_data):
    import numpy as np

    from abcfold.output.utils import get_gap_indicies

    assert get_gap_indicies() == []
    cif_path = Path(test_data.test_alphafold3_6BJ9_) / "seed-1_sample-0" / "model.cif"
    params_path = Path(test_data.test_alphafold3_6BJ9_) / "6bj9_data.json"
    cif = CifFile(cif_path, json.loads(params_path.read_text()))
    n = sum(cif.chain_lengths(mode="residues", ligand_atoms=True).values())
    mat = np.zeros((n, n), dtype=float)
    pae = Af3Pae.from_matrix(mat, cif)
    assert len(pae.scores["pae"]) == n
    assert len(pae.scores["token_chain_ids"]) == n

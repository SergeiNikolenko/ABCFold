import json
import tempfile
from pathlib import Path

from abcfold.esmfold2.af3_to_esmfold2 import Esmfold2json
from abcfold.intellifold.af3_to_intellifold import Intellifoldjson
from abcfold.opendde.af3_to_opendde import OpenDDEjson

# flake8: noqa


def test_af3_to_opendde(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        opendde_json = OpenDDEjson(temp_dir)

        data = opendde_json.json_to_json(test_data.test_inputAB_json)

        reference = {
            "name": "2PV7",
            "modelSeeds": [1],
            "sequences": [
                {"proteinChain": {"sequence": "GMRES", "count": 2,
                                  "id": ["A", "B"]}},
                {"proteinChain": {"sequence": "YANEN", "count": 1,
                                  "id": ["C"]}},
                {"ligand": {"ligand": "CCD_ATP", "count": 2,
                            "id": ["D", "E"]}},
                {"ligand": {"ligand": "CC(=O)OC1C[NH+]2CCC1CC2", "count": 1,
                            "id": ["F"]}},
            ],
        }

        assert data == reference
        assert opendde_json.seeds == [1]
        assert opendde_json.uses_msa is False


def test_af3_to_opendde_ccd_prefix(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        opendde_json = OpenDDEjson(temp_dir)
        data = opendde_json.json_to_json(test_data.test_inputLIG_json)

        ligands = [
            entry["ligand"]["ligand"]
            for entry in data["sequences"]
            if "ligand" in entry
        ]
        # CCD codes gain the OpenDDE prefix, SMILES pass through untouched
        assert "CCD_MG" in ligands
        assert "CCCCCCCCCCCC(O)=O" in ligands


def test_af3_to_opendde_msa_written(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        opendde_json = OpenDDEjson(temp_dir)
        data = opendde_json.json_to_json(test_data.test_inputAmsa_json)

        assert opendde_json.uses_msa is True
        chain = data["sequences"][0]["proteinChain"]
        msa_path = Path(chain["unpairedMsaPath"])
        assert msa_path.exists()
        assert msa_path.read_text().startswith(">")


def test_af3_to_opendde_ptm(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        opendde_json = OpenDDEjson(temp_dir)
        data = opendde_json.json_to_json(test_data.test_inputPTM_json)

        protein = data["sequences"][0]["proteinChain"]
        rna = data["sequences"][1]["rnaSequence"]
        # AF3 numbering is 1-based and is kept, the CCD codes gain their prefix
        assert protein["modifications"] == [
            {"ptmType": "CCD_HY3", "ptmPosition": 1},
            {"ptmType": "CCD_P1L", "ptmPosition": 5},
        ]
        assert rna["modifications"] == [
            {"modificationType": "CCD_2MG", "basePosition": 1},
            {"modificationType": "CCD_5MC", "basePosition": 4},
        ]


def test_af3_to_opendde_write_json(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        opendde_json = OpenDDEjson(temp_dir)
        opendde_json.json_to_json(test_data.test_inputAB_json)

        out_file = Path(temp_dir) / "opendde.json"
        opendde_json.write_json(out_file)

        with open(out_file) as f:
            written = json.load(f)

        # OpenDDE always reads a list of jobs
        assert isinstance(written, list)
        assert written[0]["name"] == "2PV7"


def test_af3_to_intellifold(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        intellifold_json = Intellifoldjson(temp_dir)

        data = intellifold_json.json_to_json(test_data.test_inputAB_json)

        with open(test_data.test_inputAB_json) as f:
            original = json.load(f)

        assert data["dialect"] == "alphafold3"
        assert data["version"] >= 2
        assert data["name"] == "2PV7"
        assert intellifold_json.seeds == [1]
        assert intellifold_json.uses_msa is False
        # IntelliFold reads the AlphaFold3 dialect, so entities are passed
        # through. Its data pipeline is always off, and AlphaFold3 then needs
        # explicit MSA and template fields: a chain without an alignment is
        # folded from its single sequence.
        for converted, source in zip(data["sequences"], original["sequences"]):
            if "protein" in converted:
                protein = converted["protein"]
                assert protein.pop("unpairedMsa") == ""
                assert protein.pop("pairedMsa") == ""
                assert protein.pop("templates") == []
            assert converted == source


def test_af3_to_intellifold_msa(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        intellifold_json = Intellifoldjson(temp_dir)
        data = intellifold_json.json_to_json(test_data.test_inputAmsa_json)

        assert intellifold_json.uses_msa is True
        protein = data["sequences"][0]["protein"]
        assert protein["unpairedMsa"].startswith(">")
        # IntelliFold needs both MSA keys present when the MSA is given by hand
        assert protein["pairedMsa"] == ""
        assert protein["templates"] == []


def test_af3_to_intellifold_does_not_mutate_input(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        with open(test_data.test_inputAB_json) as f:
            original = json.load(f)
        input_dict = json.loads(json.dumps(original))

        Intellifoldjson(temp_dir).json_to_json(input_dict)

        assert input_dict == original


def test_af3_to_esmfold2(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        esmfold2_json = Esmfold2json(temp_dir)

        data = esmfold2_json.json_to_json(test_data.test_inputAB_json)

        reference = {
            "name": "2PV7",
            "seeds": [1],
            "sequences": [
                {"type": "protein", "id": ["A", "B"], "sequence": "GMRES"},
                {"type": "protein", "id": ["C"], "sequence": "YANEN"},
                {"type": "ligand", "id": ["D", "E"], "ccd": ["ATP"]},
                {"type": "ligand", "id": ["F"],
                 "smiles": "CC(=O)OC1C[NH+]2CCC1CC2"},
            ],
        }

        assert data == reference


def test_af3_to_esmfold2_ptm_is_zero_based(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        esmfold2_json = Esmfold2json(temp_dir)
        data = esmfold2_json.json_to_json(test_data.test_inputPTM_json)

        # ESMFold2 Modification.position is 0-based, AF3 ptmPosition is 1-based
        assert data["sequences"][0]["modifications"] == [
            {"position": 0, "ccd": "HY3"},
            {"position": 4, "ccd": "P1L"},
        ]
        assert data["sequences"][1]["modifications"] == [
            {"position": 0, "ccd": "2MG"},
            {"position": 3, "ccd": "5MC"},
        ]


def test_af3_to_esmfold2_msa(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        esmfold2_json = Esmfold2json(temp_dir)
        data = esmfold2_json.json_to_json(test_data.test_inputAmsa_json)

        assert esmfold2_json.uses_msa is True
        msa_path = Path(data["sequences"][0]["msa_path"])
        assert msa_path.exists()
        assert msa_path.read_text().startswith(">")


def test_intellifold_command_flags():
    from abcfold.intellifold.run_intellifold import \
        generate_intellifold_command

    cmd = generate_intellifold_command(
        "in.json", "out", "/weights", number_of_models=3, num_recycles=7
    )

    # Flags after the bare "--" go to the vendored AlphaFold3 engine
    engine_flags = cmd[cmd.index("--") + 1:]
    assert "--num_diffusion_samples=3" in engine_flags
    assert "--num_recycles=7" in engine_flags
    # The engine's own search would need the AlphaFold3 databases
    assert "--norun_data_pipeline" in engine_flags
    assert "--force_output_dir" in engine_flags


def test_opendde_command_flags():
    from abcfold.opendde.run_opendde import generate_opendde_command

    cmd = generate_opendde_command(
        "in.json", "out", {"opendde_model": None}, number_of_models=3,
        seeds=[1, 2], use_msa=False, num_recycles=7,
    )

    def value(flag):
        return cmd[cmd.index(flag) + 1]

    assert value("-n") == "opendde_v1"
    assert value("--seeds") == "1,2"
    assert value("--sample") == "3"
    assert value("--cycle") == "7"
    # --use_msa defaults to true and would start a ColabFold search
    assert value("--use_msa") == "false"


def test_cuda_env_follows_the_inherited_mask():
    from abcfold.gpus import cuda_env

    environ = {"CUDA_VISIBLE_DEVICES": "2,5"}

    assert cuda_env("1", environ) == {"CUDA_VISIBLE_DEVICES": "5"}
    assert cuda_env("all", environ) == {}
    assert cuda_env(None, environ) == {}

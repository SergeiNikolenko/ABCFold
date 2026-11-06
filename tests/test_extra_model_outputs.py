"""Output collection for IntelliFold, OpenDDE and ESMFold2.

Each layout below is the one the backend writes, taken from its own source:
IntelliFold 2.0.4 ``run_jax_inference.write_outputs`` (vendored AlphaFold 3
v3.0.3), OpenDDE 1.2.0 ``runner/dumper.py``, and abcfold's ESMFold2 runner
script. The structures and PAE matrices are the AlphaFold 3 6BJ9 fixture, whose
B-factors are on the same 0-100 pLDDT scale that ESMFold2's
``MolecularComplex.to_mmcif`` writes.
"""

import json
import shutil
from pathlib import Path

import pytest

from abcfold.html.html_utils import get_all_cif_files
from abcfold.output.esmfold2 import Esmfold2Output
from abcfold.output.intellifold import IntellifoldOutput
from abcfold.output.opendde import OpenDDEOutput

# flake8: noqa

AF3_DIR = Path(__file__).parent / "test_data" / "alphafold3_6BJ9"
JOB = "6bj9"
SEEDS = (1, 2)
SAMPLES = (0, 1)


@pytest.fixture
def af3_sample():
    sample_dir = AF3_DIR / "seed-1_sample-0"
    confidences = json.loads((sample_dir / "confidences.json").read_text())
    return {
        "cif": sample_dir / "model.cif",
        "summary": sample_dir / "summary_confidences.json",
        "confidences": sample_dir / "confidences.json",
        "pae": confidences["pae"],
    }


@pytest.fixture
def input_params():
    return json.loads((AF3_DIR / "6bj9_data.json").read_text())


def write_intellifold(root: Path, sample: dict) -> Path:
    job_dir = root / "intellifold_results" / JOB
    for seed in SEEDS:
        for idx in SAMPLES:
            sample_dir = job_dir / f"seed-{seed}_sample-{idx}"
            sample_dir.mkdir(parents=True)
            stem = f"{JOB}_seed-{seed}_sample-{idx}"
            shutil.copy(sample["cif"], sample_dir / f"{stem}_model.cif")
            shutil.copy(sample["summary"],
                        sample_dir / f"{stem}_summary_confidences.json")
            shutil.copy(sample["confidences"],
                        sample_dir / f"{stem}_confidences.json")
    # Top-level best model and ranking table must not be taken for samples
    shutil.copy(sample["cif"], job_dir / f"{JOB}_model.cif")
    shutil.copy(sample["summary"], job_dir / f"{JOB}_summary_confidences.json")
    (job_dir / f"{JOB}_ranking_scores.csv").write_text("seed,sample,ranking_score\n")
    return root / "intellifold_results"


def write_opendde(root: Path, sample: dict, skip_full_data=()) -> Path:
    for seed in SEEDS:
        predictions = root / "opendde_results" / JOB / f"seed_{seed}" / "predictions"
        predictions.mkdir(parents=True)
        for idx in SAMPLES:
            shutil.copy(sample["cif"], predictions / f"{JOB}_sample_{idx}.cif")
            (predictions / f"{JOB}_summary_confidence_sample_{idx}.json").write_text(
                json.dumps({"ranking_score": 0.9 - idx / 10, "ptm": 0.8})
            )
            if (seed, idx) in skip_full_data:
                continue
            (predictions / f"{JOB}_full_data_sample_{idx}.json").write_text(
                json.dumps({"token_pair_pae": sample["pae"], "atom_plddt": [1.0]})
            )
    return root / "opendde_results"


def write_esmfold2(root: Path, sample: dict) -> Path:
    results = root / "esmfold2_results"
    for seed in SEEDS:
        seed_dir = results / f"esmfold2_results_seed-{seed}"
        seed_dir.mkdir(parents=True)
        for idx in SAMPLES:
            stem = f"{JOB}_seed-{seed}_sample-{idx}"
            shutil.copy(sample["cif"], seed_dir / f"{stem}_model.cif")
            (seed_dir / f"{stem}_summary_confidences.json").write_text(
                json.dumps({"ptm": 0.8, "iptm": 0.7, "ranking_score": 0.72})
            )
            (seed_dir / f"{stem}_confidences.json").write_text(
                json.dumps({"pae": sample["pae"]})
            )
    (results / f"{JOB}_ranking_scores.csv").write_text("seed,sample,ranking_score\n")
    return results


def assert_every_seed_and_sample(output):
    assert sorted(output.seeds) == ["1", "2"]
    for seed in output.seeds:
        assert sorted(output.output[seed]) == list(SAMPLES)
        assert len(output.cif_files[seed]) == len(SAMPLES)
        assert len(output.af3_pae_files[seed]) == len(SAMPLES)
        for number, files in output.output[seed].items():
            assert files["cif"].name == f"{output.label}_{seed}_{number}"
            assert files["af3_pae"].pathway.exists()
            assert files["scores"] is not None


def test_intellifold_output(tmp_path, af3_sample, input_params):
    results = write_intellifold(tmp_path, af3_sample)

    output = IntellifoldOutput([results], input_params, "6BJ9")

    assert_every_seed_and_sample(output)


def test_opendde_output_keeps_full_data(tmp_path, af3_sample, input_params):
    results = write_opendde(tmp_path, af3_sample)

    output = OpenDDEOutput([results], input_params, "6BJ9")

    assert_every_seed_and_sample(output)
    full_data = next((tmp_path / "opendde_6BJ9").rglob("*_full_data_sample_0.json"))
    # The AlphaFold3 style PAE goes next to OpenDDE's file, which keeps its
    # atom-level confidences
    assert "atom_plddt" in json.loads(full_data.read_text())


def test_opendde_missing_pae_does_not_shift_models(tmp_path, af3_sample,
                                                   input_params):
    results = write_opendde(tmp_path, af3_sample, skip_full_data={(1, 0)})

    output = OpenDDEOutput([results], input_params, "6BJ9")

    assert sorted(output.output["1"]) == [1]
    kept = output.output["1"][1]
    assert "sample_1" in kept["af3_pae"].pathway.name
    assert "sample_1" in kept["cif"].pathway.name
    assert sorted(output.output["2"]) == list(SAMPLES)


def test_esmfold2_output_keeps_seeds_apart(tmp_path, af3_sample, input_params):
    results = write_esmfold2(tmp_path, af3_sample)

    output = Esmfold2Output([results], input_params, "6BJ9")

    assert_every_seed_and_sample(output)


def test_esmfold2_plddt_stays_on_0_100(tmp_path, af3_sample, input_params):
    results = write_esmfold2(tmp_path, af3_sample)

    output = Esmfold2Output([results], input_params, "6BJ9")

    plddts = [
        value
        for cif in output.cif_files["1"]
        for value in cif.residue_plddts
        if value is not None
    ]
    assert plddts and max(plddts) <= 100


def test_save_input_moves_only_own_files(tmp_path, af3_sample, input_params):
    results = write_opendde(tmp_path, af3_sample)
    run_json = tmp_path / "abc_6bj9.json"
    run_json.write_text("{}")
    saved = tmp_path / "opendde_input"
    saved.mkdir()
    (saved / "abc_6bj9_opendde.json").write_text("[]")

    OpenDDEOutput([results], input_params, "6BJ9", save_input=True)

    # abc_<input>.json is what every later backend reads
    assert run_json.exists()
    assert (tmp_path / "opendde_6BJ9" / "input" / "abc_6bj9_opendde.json").exists()


def test_new_backends_reach_the_results_page(tmp_path, af3_sample, input_params):
    intellifold_root = tmp_path / "a"
    opendde_root = tmp_path / "b"
    esmfold2_root = tmp_path / "c"
    outputs = [
        IntellifoldOutput([write_intellifold(intellifold_root, af3_sample)],
                          input_params, "6BJ9"),
        OpenDDEOutput([write_opendde(opendde_root, af3_sample)],
                      input_params, "6BJ9"),
        Esmfold2Output([write_esmfold2(esmfold2_root, af3_sample)],
                       input_params, "6BJ9"),
    ]

    cif_files = get_all_cif_files(outputs)

    assert {name: len(models) for name, models in cif_files.items()} == {
        "IntelliFold": 4,
        "OpenDDE": 4,
        "ESMFold2": 4,
    }


def test_new_backends_get_pae_plots(tmp_path, af3_sample, input_params):
    from abcfold.plots.pae_plot import create_pae_plots

    outputs = [
        IntellifoldOutput([write_intellifold(tmp_path / "a", af3_sample)],
                          input_params, "6BJ9"),
        OpenDDEOutput([write_opendde(tmp_path / "b", af3_sample)],
                      input_params, "6BJ9"),
        Esmfold2Output([write_esmfold2(tmp_path / "c", af3_sample)],
                       input_params, "6BJ9"),
    ]

    plot_pathways = create_pae_plots(outputs, output_dir=tmp_path / "plots")

    # One plot per model: 3 backends x 2 seeds x 2 samples
    assert len(plot_pathways) == 12
    assert all(Path(plot).exists() for plot in plot_pathways.values())

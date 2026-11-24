import json
import tempfile
from pathlib import Path

import pytest

import abcfold.abcfold as abc
from abcfold import cli
from abcfold.protenix.af3_to_protenix import ProtenixJson
from abcfold.report import validate_report


def run_cli(argv):
    return abc.main(argv)


def test_failed_backend_is_reported_and_cli_survives(test_data, tmp_path,
                                                     monkeypatch):
    """Defect 8: a failed backend was reported as success / crashed the CLI."""
    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(abc, "_run_boltz", boom)
    monkeypatch.setattr(abc, "_run_chai",
                        lambda *a, **k: abc.EngineOutcome(False))
    out = tmp_path / "out"
    code = run_cli([test_data.test_inputAB_json, str(out), "-b", "-c",
                    "--no_visuals"])
    assert code == abc.EXIT_FAILED
    report = json.loads((out / "report.json").read_text())
    assert validate_report(report) == []
    engines = {e["engine"]: e for e in report["engines"]}
    assert engines["boltz"]["status"] == "failed"
    assert "CUDA out of memory" in engines["boltz"]["error"]
    assert engines["chai1"]["status"] == "failed"
    assert engines["protenix"]["status"] == "skipped"
    assert report["status"] == "failed"


def test_partial_success_exit_code(test_data, tmp_path, monkeypatch):
    monkeypatch.setattr(abc, "_run_boltz", lambda *a, **k: abc.EngineOutcome(True))
    monkeypatch.setattr(abc, "_run_chai", lambda *a, **k: abc.EngineOutcome(False))
    code = run_cli([test_data.test_inputAB_json, str(tmp_path / "o"), "-b", "-c",
                    "--no_visuals"])
    assert code == abc.EXIT_PARTIAL


def test_msa_is_kept_in_abc_input(test_data, tmp_path, monkeypatch):
    """Defect 4: the MSA must stay in the written abc_<input>.json."""
    seen = {}

    def fake_boltz(args, rt_config, run_json, input_params, name):
        seen["json"] = json.loads(Path(run_json).read_text())
        return abc.EngineOutcome(True)

    monkeypatch.setattr(abc, "_run_boltz", fake_boltz)
    msa = tmp_path / "a.a3m"
    data = json.loads(Path(test_data.test_inputA_json).read_text())
    seq = data["sequences"][0]["protein"]["sequence"]
    msa.write_text(f">query\n{seq}\n" + "".join(
        f">h{i}\n{seq[:-1]}{'ACDEFGHIKLMNPQ'[i]}\n" for i in range(12)))
    out = tmp_path / "o"
    code = run_cli([test_data.test_inputA_json, str(out), "-b", "--no_visuals",
                    "--msa", "custom", "--msa_a3m", str(msa)])
    assert code == abc.EXIT_OK
    written = json.loads((out / f"abc_{Path(test_data.test_inputA_json).name}")
                         .read_text())
    protein = written["sequences"][0]["protein"]
    assert protein["unpairedMsa"].startswith(">query")
    assert seen["json"] == written
    stats = json.loads((out / "msa_stats.json").read_text())
    assert stats["chains"]["A"]["depth"] == 13
    report = json.loads((out / "report.json").read_text())
    assert report["msa"]["source"] == "custom"


def test_shallow_msa_fails_every_engine(test_data, tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(abc, "_run_boltz",
                        lambda *a, **k: called.append(1) or abc.EngineOutcome(True))
    data = json.loads(Path(test_data.test_inputA_json).read_text())
    seq = data["sequences"][0]["protein"]["sequence"]
    msa = tmp_path / "a.a3m"
    msa.write_text(f">query\n{seq}\n")
    out = tmp_path / "o"
    code = run_cli([test_data.test_inputA_json, str(out), "-b", "--no_visuals",
                    "--msa", "custom", "--msa_a3m", str(msa)])
    assert code == abc.EXIT_FAILED
    assert called == []
    report = json.loads((out / "report.json").read_text())
    assert "too shallow" in report["engines"][1]["error"]


def test_cli_routes_legacy_invocation(monkeypatch):
    seen = {}
    monkeypatch.setattr("abcfold.abcfold.main",
                        lambda argv: seen.setdefault("argv", argv) and 0)
    cli.main(["in.json", "out", "-b"])
    assert seen["argv"] == ["in.json", "out", "-b"]
    seen.clear()
    cli.main(["run", "in.json", "out"])
    assert seen["argv"] == ["in.json", "out"]


def test_cli_envs_show(capsys):
    assert cli.main(["envs", "show", "protenix"]) == 0
    assert "protenix==2.0.0" in capsys.readouterr().out


def test_diffusion_steps_flag_aliases(test_data, tmp_path):
    parser = abc.build_run_parser()
    args = parser.parse_args([test_data.test_inputA_json, str(tmp_path),
                              "--sampling_steps", "50"])
    assert args.diffusion_steps == 50
    args = parser.parse_args([test_data.test_inputA_json, str(tmp_path)])
    assert args.diffusion_steps is None
    assert args.openfold_fetch_templates is False


def test_protenix_ions_are_ligands():
    """Defect 7: ions must be {"ligand": {"ligand": "CCD_MG", "count": N}}."""
    params = {"name": "ions", "modelSeeds": [1], "sequences": [
        {"protein": {"id": "A", "sequence": "MKV"}},
        {"ligand": {"id": ["B", "C"], "ccdCodes": ["MG"]}},
        {"ligand": {"id": "D", "ccdCodes": ["ZN"]}},
    ]}
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.json"
        src.write_text(json.dumps(params))
        data = ProtenixJson(tmp).json_to_json(src)
    ligands = [s for s in data["sequences"] if "ligand" in s]
    assert {"ligand": {"ligand": "CCD_MG", "count": 2}} in ligands
    assert {"ligand": {"ligand": "CCD_ZN", "count": 1}} in ligands
    assert not any("ion" in s for s in data["sequences"])


@pytest.mark.parametrize("flags,expected", [
    ([], False),
    (["--openfold_fetch_templates"], True),
])
def test_openfold_template_fetch_flag(test_data, tmp_path, flags, expected):
    """Defect 6: OpenFold3 must not fetch RCSB templates unless asked."""
    from abcfold.openfold3.af3_to_openfold3 import OpenfoldJson

    args = abc.build_run_parser().parse_args(
        [test_data.test_inputA_json, str(tmp_path)] + flags)
    out = tmp_path / "runner.yaml"
    OpenfoldJson(tmp_path).write_yaml(
        out, fetch_templates=args.openfold_fetch_templates
    )
    text = out.read_text()
    assert f"fetch_missing_structures: {str(expected).lower()}" in text.lower()


def test_max_msa_seqs_caps_depth_and_guard(test_data, tmp_path, monkeypatch):
    seen = {}

    def fake_boltz(args, rt_config, run_json, input_params, name):
        seen["params"] = json.loads(Path(run_json).read_text())
        return abc.EngineOutcome(True)

    monkeypatch.setattr(abc, "_run_boltz", fake_boltz)
    msa = tmp_path / "a.a3m"
    data = json.loads(Path(test_data.test_inputA_json).read_text())
    seq = data["sequences"][0]["protein"]["sequence"]
    msa.write_text(
        f">query\n{seq}\n"
        + "".join(f">h{i}\n{seq[:-1]}{'ACDEFGHIKLMNPQ'[i]}\n" for i in range(12))
    )
    out = tmp_path / "out"
    # max_msa_seqs=3 is below default min_msa_rows=10, must NOT trigger ShallowMsaError
    code = run_cli([
        test_data.test_inputA_json,
        str(out),
        "-b",
        "--no_visuals",
        "--msa",
        "custom",
        "--msa_a3m",
        str(msa),
        "--max_msa_seqs",
        "3",
    ])
    assert code == abc.EXIT_OK
    assert seen["params"]["sequences"][0]["protein"]["unpairedMsa"].count(">") == 3


def test_dry_run_skips_output_dir_and_inference(test_data, tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(
        "abcfold.dry_run.setup_environments",
        lambda args, settings: called.append(args.boltz) or ["Boltz"],
    )
    out = tmp_path / "should_not_exist"
    code = run_cli([test_data.test_inputA_json, str(out), "-b", "--dry_run"])
    assert code == abc.EXIT_OK
    assert called == [True]
    assert not out.exists()


def test_ligand_smiles_and_ccd_are_mutually_exclusive(test_data, tmp_path):
    with pytest.raises(SystemExit):
        run_cli([
            test_data.test_inputA_json,
            str(tmp_path / "out"),
            "-b",
            "--ligand_smiles",
            "CCO",
            "--ligand_ccd",
            "ATP",
        ])

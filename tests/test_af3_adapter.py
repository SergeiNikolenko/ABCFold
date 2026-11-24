import json
import os
import time

import pytest

from abcfold.alphafold3.af3_adapter import (Af3Config, af3_subprocess_env,
                                            build_af3_command,
                                            find_af3_job_dir, parse_af3_output,
                                            prepare_inference_input, run_af3,
                                            sanitised_name)


@pytest.fixture
def cfg(tmp_path):
    return Af3Config(
        runner="docker",
        image="alphafold3",
        model_dir=str(tmp_path / "models"),
        db_dir=str(tmp_path / "db"),
        gpus="1",
        num_diffusion_samples=2,
        num_recycles=3,
    )


def test_sanitised_name():
    # Same rule as AF3: lowercase, spaces -> _, drop chars outside [a-z0-9_-.]
    assert sanitised_name("KRAS G12D (MRTX1133)") == "kras_g12d_mrtx1133"


def test_docker_full_command(tmp_path, cfg):
    cmd = build_af3_command(tmp_path / "in" / "job.json", tmp_path / "out", cfg)
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert cmd[cmd.index("--gpus") + 1] == "device=1"
    assert f"{tmp_path / 'models'}:/root/models:ro" in cmd
    assert f"{tmp_path / 'db'}:/root/public_databases:ro" in cmd
    i = cmd.index("alphafold3")
    assert cmd[i + 1:i + 3] == ["python", "run_alphafold.py"]
    assert "--json_path=/root/af_input/job.json" in cmd
    assert "--db_dir=/root/public_databases" in cmd
    assert "--num_diffusion_samples=2" in cmd
    assert "--num_recycles=3" in cmd
    assert not any("run_data_pipeline" in c or "run_inference" in c for c in cmd)


def test_docker_data_pipeline_only(tmp_path, cfg):
    cfg.model_dir = None
    cmd = build_af3_command(tmp_path / "job.json", tmp_path / "out", cfg,
                            mode="data_pipeline")
    assert "--run_inference=false" in cmd
    assert "--gpus" not in cmd
    assert not any(c.startswith("--model_dir") for c in cmd)
    assert not any("/root/models" in c for c in cmd)


def test_docker_inference_only_needs_no_databases(tmp_path, cfg):
    cfg.db_dir = None
    cmd = build_af3_command(tmp_path / "job.json", tmp_path / "out", cfg,
                            mode="inference")
    assert "--run_data_pipeline=false" in cmd
    assert not any("public_databases" in c for c in cmd)
    assert "--model_dir=/root/models" in cmd


def test_missing_paths_raise(tmp_path, cfg):
    cfg.db_dir = None
    with pytest.raises(ValueError, match="database"):
        build_af3_command(tmp_path / "j.json", tmp_path, cfg, mode="full")


def test_apptainer_command(tmp_path, cfg):
    cfg.runner = "apptainer"
    cfg.image = str(tmp_path / "af3.sif")
    cmd = build_af3_command(tmp_path / "job.json", tmp_path / "out", cfg)
    assert cmd[:3] == ["apptainer", "exec", "--nv"]
    assert "--bind" in cmd
    assert cmd[cmd.index(cfg.image) + 2] == "/app/alphafold/run_alphafold.py"
    env = af3_subprocess_env(cfg, environ={"LD_PRELOAD": "x"})
    assert env["APPTAINERENV_CUDA_VISIBLE_DEVICES"] == "1"
    assert "LD_PRELOAD" not in env


def test_wrapper_command_template(tmp_path, cfg):
    """Site wrapper such as lobachevsky's ~/bin/alphafold3 <json> <out> [gpu]."""
    cfg.runner = "command"
    cfg.command = "~/bin/alphafold3 {input_json} {output_dir} {gpus}"
    cfg.append_flags = False
    cmd = build_af3_command(tmp_path / "job.json", tmp_path / "out", cfg)
    assert cmd == [os.path.expanduser("~/bin/alphafold3"),
                   str(tmp_path / "job.json"), str(tmp_path / "out"), "1"]
    cfg.append_flags = True
    cfg.gpus = "all"
    cmd = build_af3_command(tmp_path / "job.json", tmp_path / "out", cfg,
                            mode="inference")
    assert cmd[3:] == ["--run_data_pipeline=false", "--num_diffusion_samples=2",
                       "--num_recycles=3"]


def test_from_runtime_config():
    cfg = Af3Config.from_runtime_config(
        {"af3_runner": "command", "af3_command": "wrap {input_json}",
         "af3_append_flags": "false", "model_params": "None"},
        gpus="2",
    )
    assert cfg.runner == "command"
    assert cfg.append_flags is False
    assert cfg.model_dir is None
    assert cfg.gpus == "2"


def test_prepare_inference_input_fills_msa_fields():
    params = {"sequences": [
        {"protein": {"id": "A", "sequence": "MK"}},
        {"protein": {"id": "B", "sequence": "GS", "unpairedMsa": ">q\nGS\n"}},
        {"rna": {"id": "C", "sequence": "AGCU"}},
        {"ligand": {"id": "D", "ccdCodes": ["ATP"]}},
    ]}
    out = prepare_inference_input(params)
    a, b, c = (out["sequences"][i] for i in range(3))
    assert a["protein"] == {"id": "A", "sequence": "MK", "unpairedMsa": "",
                            "pairedMsa": "", "templates": []}
    assert b["protein"]["unpairedMsa"] == ">q\nGS\n"
    assert c["rna"]["unpairedMsa"] == ""
    assert "unpairedMsa" not in params["sequences"][0]["protein"]


def make_af3_output(root, name="kras", samples=((1, 0), (1, 1)), prefixed=True):
    job = root / name
    job.mkdir(parents=True)
    (job / f"{name}_model.cif").write_text("data_top\n")
    (job / f"{name}_summary_confidences.json").write_text('{"ranking_score": 0.9}')
    (job / f"{name}_confidences.json").write_text('{"pae": []}')
    (job / f"{name}_data.json").write_text("{}")
    rows = ["seed,sample,ranking_score"]
    for seed, sample in samples:
        d = job / f"seed-{seed}_sample-{sample}"
        d.mkdir()
        pre = f"{name}_seed-{seed}_sample-{sample}_" if prefixed else ""
        (d / f"{pre}model.cif").write_text("data_x\n")
        (d / f"{pre}summary_confidences.json").write_text(
            json.dumps({"ranking_score": 0.5 + sample / 10}))
        (d / f"{pre}confidences.json").write_text('{"pae": []}')
        rows.append(f"{seed},{sample},{0.5 + sample / 10}")
    (job / f"{name}_ranking_scores.csv").write_text("\n".join(rows) + "\n")
    return job


@pytest.mark.parametrize("prefixed", [True, False])
def test_parse_af3_output(tmp_path, prefixed):
    job = make_af3_output(tmp_path, prefixed=prefixed)
    result = parse_af3_output(tmp_path, "KRAS")
    assert result.job_dir == job
    assert result.top_model.name == "kras_model.cif"
    assert result.top_summary.name == "kras_summary_confidences.json"
    assert result.top_confidences.name == "kras_confidences.json"
    assert result.data_json.name == "kras_data.json"
    assert [(s.seed, s.sample) for s in result.samples] == [(1, 0), (1, 1)]
    assert result.samples[1].ranking_score == pytest.approx(0.6)
    assert "summary" not in result.samples[0].confidences.name


def test_find_timestamped_job_dir(tmp_path):
    make_af3_output(tmp_path)
    time.sleep(0.01)
    newer = tmp_path / "kras_20261016_101010"
    newer.mkdir()
    (tmp_path / "kras_other").mkdir()
    assert find_af3_job_dir(tmp_path, "kras") == newer
    assert parse_af3_output(tmp_path, "nothing") is None


def test_run_af3_with_fake_wrapper(tmp_path, monkeypatch, cfg):
    calls = []

    def fake_run(argv, stdout, stderr, env, check):
        calls.append(argv)
        make_af3_output(tmp_path / "out")

        class P:
            returncode = 0
        return P()

    monkeypatch.setattr("abcfold.alphafold3.af3_adapter.subprocess.run", fake_run)
    params = {"name": "kras", "sequences": [{"protein": {"id": "A",
                                                         "sequence": "MK"}}]}
    result = run_af3(params, tmp_path / "out", cfg, mode="inference")
    assert result.ok
    written = json.loads((tmp_path / "out/af3_input/kras.json").read_text())
    assert written["sequences"][0]["protein"]["unpairedMsa"] == ""
    assert "--run_data_pipeline=false" in calls[0]


def test_run_af3_failure_raises(tmp_path, monkeypatch, cfg):
    def fake_run(argv, stdout, stderr, env, check):
        stdout.write("boom\n")

        class P:
            returncode = 3
        return P()

    monkeypatch.setattr("abcfold.alphafold3.af3_adapter.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="exited with 3"):
        run_af3({"name": "x", "sequences": []}, tmp_path / "o", cfg)

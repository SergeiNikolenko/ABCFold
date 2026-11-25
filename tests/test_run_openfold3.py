import json
import shutil
from pathlib import Path

import abcfold.openfold3.run_openfold3 as rof
from abcfold.output.openfold3 import OpenfoldOutput


def test_run_openfold_invokes_once_for_multiple_seeds(tmp_path, monkeypatch):
    inp = tmp_path / "multi.json"
    inp.write_text(
        json.dumps(
            {
                "name": "multi",
                "modelSeeds": [1, 2, 3],
                "sequences": [{"protein": {"id": ["A"], "sequence": "GMRES"}}],
            }
        )
    )
    out = tmp_path / "out"
    calls = []

    class FakeEnv:
        def run(self, cmd, extra_env=None):
            calls.append(cmd)
            out_dir = Path(cmd[cmd.index("--output_dir") + 1])
            sample = out_dir / "multi" / "seed_1" / "multi_seed_1_sample_1_model.cif"
            sample.parent.mkdir(parents=True, exist_ok=True)
            sample.write_text("data_test\n")

    monkeypatch.setattr(rof, "ensure_openfold_env", lambda config: FakeEnv())
    monkeypatch.setattr(rof, "ensure_openfold_checkpoint", lambda *a, **k: None)

    ok = rof.run_openfold(
        input_json=inp,
        output_dir=out,
        config={"openfold_weights": str(tmp_path / "ckpt")},
        save_input=True,
    )
    assert ok is True
    assert len(calls) == 1
    out_dir_arg = Path(calls[0][calls[0].index("--output_dir") + 1])
    assert out_dir_arg.name == "openfold_results"
    assert "seeds: [1, 2, 3]" in (out / "openfold3_runner.yml").read_text()


def test_openfold_output_reads_single_multiseed_layout(test_data, tmp_path):
    src = Path("tests/test_data/openfold_6BJ9_seed-1/6BJ9/seed_1")
    dst = tmp_path / "openfold_results" / "6BJ9" / "seed_42"
    shutil.copytree(src, dst)
    params = json.loads(Path(test_data.test_inputA_json).read_text())
    of_out = OpenfoldOutput([tmp_path / "openfold_results"], params, "6BJ9")
    assert "seed-42" in of_out.seeds
    assert len(of_out.cif_files["seed-42"]) == 2

import json
import shutil
import threading
import time
import types
from pathlib import Path


import abcfold.abcfold as abc
from abcfold.alphafold3.af3_adapter import (
    Af3Config,
    af3_subprocess_env,
    build_af3_command,
)
from abcfold.backend_envs import (
    clear_env_caches,
    container_binds,
    jax_cache_dir,
    jax_runtime_env,
)
from abcfold.batch import (
    BatchJob,
    _parse_batch_run_args,
    estimate_job_tokens,
    parse_models,
    run_batch,
    sort_jobs_by_length,
)
from abcfold.boltz import run_boltz as rb
from abcfold.boltz.af3_to_boltz import BoltzYaml
from abcfold.chai1.af3_to_chai import ChaiFasta
from abcfold.esmfold2 import run_esmfold2 as re2
from abcfold.intellifold.run_intellifold import (
    generate_intellifold_command,
    run_intellifold,
)
from abcfold.opendde import run_opendde as rod
from abcfold.openfold3 import run_openfold3 as rof
from abcfold.output.file_handlers import CifFile, ConfidenceJsonFile
from abcfold.output.protenix import ProtenixOutput
from abcfold.output.utils import Af3Pae
from abcfold.protenix import run_protenix as rpx


def test_af3_pae_compact_json_and_preloaded_confidence(tmp_path):
    pae = Af3Pae({"pae": [[0.1, 1.2], [1.2, 0.2]], "contact_probs": [[0.0, 0.0]]})
    out = tmp_path / "pae.json"
    pae.to_file(out)
    raw = out.read_text()
    assert "\n" not in raw
    assert ": " not in raw
    assert json.loads(raw) == pae.scores

    # Preloaded ConfidenceJsonFile does not touch disk
    missing = tmp_path / "does_not_exist.json"
    cjf = ConfidenceJsonFile(missing, data=pae.scores)
    assert cjf.data == pae.scores

    # Lazy-load from disk when data=None
    cjf_disk = ConfidenceJsonFile(out)
    assert cjf_disk.data == pae.scores


def test_cif_check_clashes_cached_and_invalidated(test_data, tmp_path):
    cif_copy = tmp_path / "6BJ9.cif"
    shutil.copy(test_data.test_6BJ9_cif, cif_copy)
    params = json.loads(Path(test_data.test_inputA_json).read_text())
    cf = CifFile(cif_copy, params)

    first = cf.check_clashes()
    second = cf.check_clashes()
    assert first is second
    assert cf.clashes == len(first[0])
    assert cf.clashes_residues == len(first[1])

    # Cache invalidates cleanly on update
    cf.update()
    assert cf._clash_cache == {}
    third = cf.check_clashes()
    assert len(third[0]) == len(first[0])
    assert len(third[1]) == len(first[1])


def test_postprocess_converters_skip_rcsb_http(monkeypatch, test_data, tmp_path):
    def _no_net(*args, **kwargs):
        raise AssertionError("Network call must not happen when create_files=False")

    monkeypatch.setattr("urllib.request.urlopen", _no_net)
    params = json.loads(Path(test_data.test_inputLIG_json).read_text())

    by = BoltzYaml(tmp_path, create_files=False)
    by.json_to_yaml(params)
    assert by.chain_ids

    cf = ChaiFasta(tmp_path, config=test_data.config_dict, create_files=False)
    cf.json_to_fasta(params)
    assert cf.chain_ids


def test_pae_plots_render_in_process_without_subprocesses(
    monkeypatch, test_data, tmp_path
):
    import multiprocessing
    import subprocess

    from abcfold.output.boltz import BoltzOutput
    from abcfold.plots.pae_plot import create_pae_plots

    def _forbid(*args, **kwargs):
        raise AssertionError("create_pae_plots must render in-process")

    monkeypatch.setattr(subprocess, "Popen", _forbid)
    monkeypatch.setattr(multiprocessing, "Process", _forbid)

    src = Path("tests/test_data/boltz_6BJ9_seed-1")
    dst = tmp_path / "boltz_results_6BJ9_seed-1"
    shutil.copytree(src, dst)
    params = json.loads(
        Path("tests/test_data/alphafold3_6BJ9/6bj9_data.json").read_text()
    )
    bo = BoltzOutput([dst], params, "6BJ9", False)
    plots_dir = tmp_path / ".plots"
    plots_dir.mkdir()
    result = create_pae_plots([bo], plots_dir)
    assert len(result) == 2
    assert all(Path(p).is_file() for p in result.values())


def test_jax_runtime_env_and_container_binds(tmp_path, monkeypatch):
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    clear_env_caches()
    cfg = {"jax_cache_dir": str(tmp_path / "jax_cache")}
    jdir = jax_cache_dir(cfg)
    assert jdir == (tmp_path / "jax_cache").resolve()
    env = jax_runtime_env(cfg)
    assert env["JAX_COMPILATION_CACHE_DIR"] == str(jdir)
    assert env["XLA_FLAGS"] == "--xla_gpu_enable_triton_gemm=false"
    assert env["XLA_CLIENT_MEM_FRACTION"] == "0.95"

    class FakeSettings:
        def get(self, key):
            return str(tmp_path / "jax_cache") if key == "jax_cache_dir" else None

    binds = container_binds(FakeSettings())
    assert str(tmp_path / "jax_cache") in binds


def test_run_boltz_persistent_worker_single_and_multi_gpu(tmp_path, monkeypatch):
    inp = tmp_path / "target.json"
    inp.write_text(
        json.dumps(
            {
                "name": "target",
                "modelSeeds": [42, 43],
                "sequences": [{"protein": {"id": "A", "sequence": "GMRES"}}],
            }
        )
    )

    calls = []

    class FakeEnv:
        def run(self, cmd, capture_output=False):
            manifest_path = Path(cmd[-1])
            spec = json.loads(manifest_path.read_text())
            calls.append((cmd, spec))
            for seed, out_dir in spec["seed_out_dirs"].items():
                cif = Path(out_dir) / "predictions" / "target" / "target_model_0.cif"
                cif.parent.mkdir(parents=True, exist_ok=True)
                cif.write_text("data_test\n")
            return "ok"

    monkeypatch.setattr(rb, "ensure_boltz_env", lambda config: FakeEnv())

    # Single GPU -> 1 persistent worker covering BOTH seeds
    out1 = tmp_path / "out_1gpu"
    ok1 = rb.run_boltz(
        inp,
        out1,
        config={"boltz_weights": str(tmp_path / "weights")},
        gpus="0",
        boltz_use_potentials=True,
    )
    assert ok1 is True
    assert len(calls) == 1
    assert calls[0][1]["seeds"] == [42, 43]
    assert calls[0][1]["use_potentials"] is True
    assert "CUDA_VISIBLE_DEVICES=0" in calls[0][0]

    # Two GPUs -> 2 parallel workers, 1 seed per GPU
    calls.clear()
    out2 = tmp_path / "out_2gpu"
    ok2 = rb.run_boltz(
        inp,
        out2,
        config={"boltz_weights": str(tmp_path / "weights")},
        gpus="0,1",
    )
    assert ok2 is True
    assert len(calls) == 2
    seen_seeds = sorted(s for _, spec in calls for s in spec["seeds"])
    assert seen_seeds == [42, 43]

    # Legacy command builder exposes --use_potentials when enabled
    cmd_pot = rb.generate_boltz_command(
        tmp_path / "x.yaml",
        tmp_path / "out",
        config={"boltz_weights": None},
        boltz_use_potentials=True,
    )
    assert "--use_potentials" in cmd_pot


def test_run_boltz_batch_persistent_worker_sharding(tmp_path, monkeypatch):
    inputs = []
    for name in ("t1", "t2"):
        p = tmp_path / f"{name}.json"
        p.write_text(
            json.dumps(
                {
                    "name": name,
                    "modelSeeds": [1, 2],
                    "sequences": [{"protein": {"id": "A", "sequence": "GMRES"}}],
                }
            )
        )
        inputs.append(p)

    calls = []

    class FakeEnv:
        def run(self, cmd, capture_output=False):
            spec = json.loads(Path(cmd[-1]).read_text())
            calls.append((cmd, spec))
            return "ok"

    monkeypatch.setattr(rb, "ensure_boltz_env", lambda config: FakeEnv())

    # 1 GPU: 1 worker across both targets & both seeds
    ok = rb.run_boltz_batch(
        inputs,
        tmp_path / "batch1",
        config={"boltz_weights": str(tmp_path / "w")},
        gpus="0",
    )
    assert ok is True
    assert len(calls) == 1
    assert calls[0][1]["seeds"] == [1, 2]
    assert sorted(calls[0][1]["seed_record_ids"]["1"]) == ["t1", "t2"]

    # 2 GPUs: 2 input shards, 1 per GPU
    calls.clear()
    ok2 = rb.run_boltz_batch(
        inputs,
        tmp_path / "batch2",
        config={"boltz_weights": str(tmp_path / "w")},
        gpus="0,1",
    )
    assert ok2 is True
    assert len(calls) == 2
    gpus_used = {
        next(tok for tok in cmd if tok.startswith("CUDA_VISIBLE_DEVICES="))
        for cmd, _ in calls
    }
    assert gpus_used == {"CUDA_VISIBLE_DEVICES=0", "CUDA_VISIBLE_DEVICES=1"}


def test_protenix_single_process_multiseed_and_output(tmp_path, monkeypatch):
    inp = tmp_path / "multi.json"
    inp.write_text(
        json.dumps(
            {
                "name": "multi",
                "modelSeeds": [42, 43],
                "sequences": [{"protein": {"id": "A", "sequence": "GMRES"}}],
            }
        )
    )
    calls = []

    class FakeEnv:
        def has_executable(self, name):
            return False

        def run(self, cmd, extra_env=None, log_file=None):
            calls.append((cmd, extra_env))
            dump_dir = Path(cmd[cmd.index("--dump_dir") + 1])
            for seed in (42, 43):
                cif = (
                    dump_dir
                    / "multi"
                    / f"seed_{seed}"
                    / "predictions"
                    / f"multi_seed_{seed}_sample_0.cif"
                )
                cif.parent.mkdir(parents=True, exist_ok=True)
                cif.write_text("data_test\n")

    monkeypatch.setattr(rpx, "ensure_protenix_env", lambda config: FakeEnv())
    ok = rpx.run_protenix(inp, tmp_path / "out", config={}, gpus="0")
    assert ok is True
    assert len(calls) == 1
    cmd = calls[0][0]
    assert cmd[cmd.index("--seeds") + 1] == "42,43"

    # ProtenixOutput reads single protenix_results dir with multiple seed_<N> dirs
    src_pred = Path("tests/test_data/protenix_6BJ9_seed-1/6BJ9/seed_1")
    out_dir = tmp_path / "protenix_results"
    shutil.copytree(src_pred, out_dir / "6BJ9" / "seed_42")
    shutil.copytree(src_pred, out_dir / "6BJ9" / "seed_43")
    params = json.loads(
        Path("tests/test_data/alphafold3_6BJ9/6bj9_data.json").read_text()
    )
    po = ProtenixOutput([out_dir], params, "6BJ9")
    assert sorted(po.seeds) == ["seed-42", "seed-43"]
    assert len(po.cif_files["seed-42"]) == 2
    assert len(po.cif_files["seed-43"]) == 2


def test_protenix_batch_combines_jobs_into_one_invocation(tmp_path, monkeypatch):
    items = []
    for name in ("p1", "p2"):
        inp = tmp_path / f"{name}.json"
        inp.write_text(
            json.dumps(
                {
                    "name": name,
                    "modelSeeds": [7],
                    "sequences": [{"protein": {"id": "A", "sequence": "GMRES"}}],
                }
            )
        )
        items.append((inp, tmp_path / f"out_{name}"))

    calls = []

    class FakeEnv:
        def has_executable(self, name):
            return False

        def run(self, cmd, extra_env=None, log_file=None):
            calls.append(cmd)
            batch_path = Path(cmd[cmd.index("--input_json_path") + 1])
            batch_json = json.loads(batch_path.read_text())
            dump_dir = Path(cmd[cmd.index("--dump_dir") + 1])
            for job in batch_json:
                cif = (
                    dump_dir
                    / job["name"]
                    / "seed_7"
                    / "predictions"
                    / f"{job['name']}_seed_7_sample_0.cif"
                )
                cif.parent.mkdir(parents=True, exist_ok=True)
                cif.write_text("data_test\n")

    monkeypatch.setattr(rpx, "ensure_protenix_env", lambda config: FakeEnv())
    res = rpx.run_protenix_batch(items, config={}, gpus="0")
    assert len(calls) == 1
    assert all(res.values())
    assert (tmp_path / "out_p1" / "protenix_results" / "p1").is_dir()
    assert (tmp_path / "out_p2" / "protenix_results" / "p2").is_dir()


def test_chai_resolve_low_memory_by_vram(monkeypatch):
    import sys

    # Provide lightweight stub for chai_lab/typer so chai.py imports without chai env
    fake_chai_lab = types.ModuleType("chai_lab")
    fake_chai1 = types.ModuleType("chai_lab.chai1")

    def fake_run_inference(fasta_file, output_dir, device=None, low_memory=True):
        return None

    fake_chai1.run_inference = fake_run_inference
    fake_typer = types.ModuleType("typer")

    monkeypatch.setitem(sys.modules, "chai_lab", fake_chai_lab)
    monkeypatch.setitem(sys.modules, "chai_lab.chai1", fake_chai1)
    monkeypatch.setitem(sys.modules, "typer", fake_typer)

    import importlib

    import abcfold.chai1.chai as chai_mod

    importlib.reload(chai_mod)

    class Props:
        def __init__(self, gb):
            self.total_memory = gb * (1024**3)

    def _fake_torch(avail: bool, gb: int):
        mod = types.ModuleType("torch")
        cuda = types.SimpleNamespace(
            is_available=lambda: avail,
            get_device_properties=lambda idx=0: Props(gb),
        )
        mod.cuda = cuda
        return mod

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, 40))
    assert chai_mod.resolve_low_memory(None, "cuda:0") is False
    assert chai_mod.resolve_low_memory(True, "cuda:0") is True
    assert chai_mod.resolve_low_memory(None, "cpu") is True

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, 16))
    assert chai_mod.resolve_low_memory(None, "cuda:0") is True
    assert chai_mod.resolve_low_memory(False, "cuda:0") is False


def test_intellifold_and_af3_jax_cache_flags(tmp_path, monkeypatch):
    cache = tmp_path / "jax_cache"
    cmd = generate_intellifold_command(
        "in.json", "out", "/weights", jax_compilation_cache_dir=cache
    )
    assert f"--jax_compilation_cache_dir={cache}" in cmd

    seen = {}

    class FakeEnv:
        def run(self, cmd, extra_env=None):
            seen["cmd"] = cmd
            seen["env"] = extra_env

    monkeypatch.setattr(
        "abcfold.intellifold.run_intellifold.ensure_intellifold_env",
        lambda config: FakeEnv(),
    )
    inp = tmp_path / "in.json"
    inp.write_text(
        json.dumps(
            {
                "name": "x",
                "modelSeeds": [1],
                "sequences": [{"protein": {"id": "A", "sequence": "GMRES"}}],
            }
        )
    )
    ok = run_intellifold(
        inp, tmp_path / "out", config={"jax_cache_dir": str(cache)}, gpus="0"
    )
    assert ok is True
    assert any(a.startswith("--jax_compilation_cache_dir=") for a in seen["cmd"])
    assert seen["env"]["XLA_FLAGS"] == "--xla_gpu_enable_triton_gemm=false"

    cfg = Af3Config(
        runner="apptainer",
        image=str(tmp_path / "af3.sif"),
        model_dir=str(tmp_path / "models"),
        db_dir=str(tmp_path / "db"),
        jax_compilation_cache_dir=str(cache),
        gpus="0",
    )
    af3_cmd = build_af3_command(inp, tmp_path / "af3_out", cfg, mode="inference")
    assert "--jax_compilation_cache_dir=/root/jax_cache" in af3_cmd
    assert f"{cache.resolve()}:/root/jax_cache" in af3_cmd
    af3_env = af3_subprocess_env(cfg, environ={})
    assert af3_env["XLA_FLAGS"] == "--xla_gpu_enable_triton_gemm=false"
    assert af3_env["APPTAINERENV_JAX_COMPILATION_CACHE_DIR"] == "/root/jax_cache"


def test_openfold_opendde_esmfold2_batch_runners(tmp_path, monkeypatch):
    items = []
    for name in ("j1", "j2"):
        inp = tmp_path / f"{name}.json"
        inp.write_text(
            json.dumps(
                {
                    "name": name,
                    "modelSeeds": [1],
                    "sequences": [{"protein": {"id": "A", "sequence": "GMRES"}}],
                }
            )
        )
        items.append((inp, tmp_path / f"out_{name}"))

    # OpenFold3 batch
    of_calls = []

    class FakeOfEnv:
        def run(self, cmd, extra_env=None):
            of_calls.append(cmd)
            qjson = json.loads(Path(cmd[cmd.index("--query_json") + 1]).read_text())
            out_dir = Path(cmd[cmd.index("--output_dir") + 1])
            for qname in qjson["queries"]:
                cif = out_dir / qname / "seed_1" / f"{qname}_seed_1_sample_1_model.cif"
                cif.parent.mkdir(parents=True, exist_ok=True)
                cif.write_text("data_test\n")

    monkeypatch.setattr(rof, "ensure_openfold_env", lambda config: FakeOfEnv())
    monkeypatch.setattr(rof, "ensure_openfold_checkpoint", lambda *a, **k: None)
    of_res = rof.run_openfold_batch(
        items, config={"openfold_weights": str(tmp_path / "of")}, gpus="0"
    )
    assert len(of_calls) == 1
    assert all(of_res.values())

    # OpenDDE batch
    od_calls = []

    class FakeOdEnv:
        def run(self, cmd, extra_env=None):
            od_calls.append(cmd)
            jobs = json.loads(Path(cmd[cmd.index("-i") + 1]).read_text())
            out_dir = Path(cmd[cmd.index("-o") + 1])
            for j in jobs:
                cif = out_dir / j["name"] / "seed_1" / "predictions" / "m_sample_0.cif"
                cif.parent.mkdir(parents=True, exist_ok=True)
                cif.write_text("data_test\n")

    monkeypatch.setattr(rod, "ensure_opendde_env", lambda config: FakeOdEnv())
    od_res = rod.run_opendde_batch(items, config={}, gpus="0")
    assert len(od_calls) == 1
    assert all(od_res.values())
    assert "--use_tfg_guidance" in rod.generate_opendde_command(
        "i.json", "o", {}, use_tfg_guidance=True
    )

    # ESMFold2 batch
    esm_calls = []

    class FakeEsmEnv:
        def run(self, cmd, extra_env=None):
            esm_calls.append(cmd)
            manifest = json.loads(Path(cmd[cmd.index("--manifest") + 1]).read_text())
            for job in manifest["jobs"]:
                cif = (
                    Path(job["output_dir"])
                    / "esmfold2_results_seed-1"
                    / "pred_seed-1_sample-0_model.cif"
                )
                cif.parent.mkdir(parents=True, exist_ok=True)
                cif.write_text("data_test\n")

    monkeypatch.setattr(re2, "ensure_esmfold2_env", lambda config: FakeEsmEnv())
    esm_res = re2.run_esmfold2_batch(items, config={}, gpus="0")
    assert len(esm_calls) == 1
    assert all(esm_res.values())


def test_abcfold_multi_gpu_dispatches_engines_in_parallel(
    test_data, tmp_path, monkeypatch
):
    barrier = threading.Barrier(2, timeout=5.0)
    seen_gpus = {}

    def fake_boltz(args, rt_config, run_json, input_params, name):
        seen_gpus["boltz"] = args.gpus
        barrier.wait()
        return abc.EngineOutcome(True)

    def fake_chai(args, rt_config, run_json, input_params, temp_dir, name):
        seen_gpus["chai1"] = args.gpus
        barrier.wait()
        return abc.EngineOutcome(True)

    monkeypatch.setattr(abc, "_run_boltz", fake_boltz)
    monkeypatch.setattr(abc, "_run_chai", fake_chai)

    out = tmp_path / "out"
    code = abc.main(
        [
            test_data.test_inputAB_json,
            str(out),
            "-b",
            "-c",
            "--gpus",
            "0,1",
            "--no_visuals",
        ]
    )
    assert code == abc.EXIT_OK
    assert set(seen_gpus.values()) == {"0", "1"}


def test_batch_length_sorting_aliases_and_phased_dispatch(tmp_path, monkeypatch):
    assert parse_models("intellifold;opendde;esmfold2") == [
        "intellifold",
        "opendde",
        "esmfold2",
    ]
    short = BatchJob(
        "short",
        {
            "name": "short",
            "modelSeeds": [1],
            "sequences": [{"protein": {"id": "A", "sequence": "M" * 20}}],
        },
        ["boltz"],
    )
    long = BatchJob(
        "long",
        {
            "name": "long",
            "modelSeeds": [1],
            "sequences": [{"protein": {"id": "A", "sequence": "M" * 200}}],
        },
        ["boltz"],
    )
    assert estimate_job_tokens(long) > estimate_job_tokens(short)
    assert [j.id for j in sort_jobs_by_length([short, long])] == ["long", "short"]

    assert (
        _parse_batch_run_args(["--num_recycles", "3", "--boltz_use_potentials"])
        is not None
    )
    assert _parse_batch_run_args(["--mmseqs2"]) is None

    # Phased batch invokes run_boltz_batch ONCE for 2 jobs when runner is None
    manifest = tmp_path / "m.csv"
    manifest.write_text(
        "id,protein,models\n"
        f"small,{'M' * 20},boltz\n"
        f"large,{'M' * 80},boltz\n"
    )
    boltz_batch_calls = []

    def fake_boltz_batch(input_jsons, stage_dir, **kwargs):
        paths = [Path(p) for p in input_jsons]
        boltz_batch_calls.append([p.stem for p in paths])
        for p in paths:
            cif = (
                Path(stage_dir)
                / "boltz_batch"
                / "seed-1"
                / "boltz_results_seed-1"
                / "predictions"
                / p.stem
                / f"{p.stem}_model_0.cif"
            )
            cif.parent.mkdir(parents=True, exist_ok=True)
            cif.write_text("data_test\n")
        return True

    monkeypatch.setattr("abcfold.boltz.run_boltz.run_boltz_batch", fake_boltz_batch)
    monkeypatch.setattr("abcfold.output.boltz.BoltzOutput", lambda *a, **k: None)

    t0 = time.monotonic()
    status = run_batch(
        manifest,
        tmp_path / "batch_out",
        gpus="0",
        msa="none",
        batched=True,
    )
    assert time.monotonic() - t0 < 10.0
    assert len(boltz_batch_calls) == 1
    # Longest job ordered first
    assert boltz_batch_calls[0] == ["abc_large", "abc_small"]
    assert status["counts"] == {"ok": 2}
    assert (tmp_path / "batch_out" / "jobs" / "large" / "report.json").exists()
    assert (tmp_path / "batch_out" / "jobs" / "small" / "report.json").exists()


def test_chai_inference_caches_and_case_local_sharding(monkeypatch, tmp_path):
    import importlib
    import sys

    from abcfold.chai1.run_chai1_fast import _write_device_manifests

    # Case-local sharding keeps all seeds of a complex on the same GPU
    tasks = [
        {"case_id": "c1", "seed": 1, "output_dir": "o1"},
        {"case_id": "c1", "seed": 2, "output_dir": "o2"},
        {"case_id": "c2", "seed": 1, "output_dir": "o3"},
        {"case_id": "c2", "seed": 2, "output_dir": "o4"},
    ]
    manifests = _write_device_manifests(
        tasks, devices=["cuda:0", "cuda:1"], work_root=tmp_path,
        number_of_models=2, num_recycles=3,
    )
    m0 = json.loads(manifests[0].read_text())
    m1 = json.loads(manifests[1].read_text())
    assert {t["case_id"] for t in m0} == {"c1"}
    assert {t["case_id"] for t in m1} == {"c2"}

    # load_exported and make_all_atom_feature_context caches deduplicate across seeds
    load_calls = []
    ctx_calls = []
    fake_chai_lab = types.ModuleType("chai_lab")
    fake_chai1 = types.ModuleType("chai_lab.chai1")

    def fake_load(comp_key, device):
        load_calls.append((comp_key, str(device)))
        return object()

    def fake_ctx(fasta_file, **kwargs):
        ctx_calls.append(str(fasta_file))
        return {"fasta": str(fasta_file)}

    def fake_run_inference(fasta_file, output_dir, **kwargs):
        return None

    fake_chai1.load_exported = fake_load
    fake_chai1.make_all_atom_feature_context = fake_ctx
    fake_chai1.run_inference = fake_run_inference
    monkeypatch.setitem(sys.modules, "chai_lab", fake_chai_lab)
    monkeypatch.setitem(sys.modules, "chai_lab.chai1", fake_chai1)
    monkeypatch.setitem(sys.modules, "typer", types.ModuleType("typer"))

    import abcfold.chai1.chai as chai_mod

    importlib.reload(chai_mod)
    chai_mod._install_inference_caches()

    m_a = fake_chai1.load_exported("trunk.pt", "cuda:0")
    m_b = fake_chai1.load_exported("trunk.pt", "cuda:0")
    assert m_a is m_b
    assert len(load_calls) == 1

    f1 = tmp_path / "a.fasta"
    f1.write_text(">protein|A\nGMRES\n")
    c_a = fake_chai1.make_all_atom_feature_context(f1, msa_directory=tmp_path)
    c_b = fake_chai1.make_all_atom_feature_context(f1, msa_directory=tmp_path)
    assert c_a is c_b
    assert len(ctx_calls) == 1

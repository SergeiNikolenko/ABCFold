import pytest

from abcfold.gpus import docker_gpu_flags, to_physical
from abcfold.openfold3.run_openfold3 import _gpu_env
from abcfold.protenix.run_protenix import protenix_runtime_env


def test_indices_follow_inherited_mask():
    """Defect 13: --gpus 0 under CUDA_VISIBLE_DEVICES=2 must stay on physical GPU 2."""
    env = {"CUDA_VISIBLE_DEVICES": "2,3"}
    assert to_physical("0", env) == "2"
    assert to_physical("1,0", env) == "3,2"
    assert to_physical("all", env) is None
    assert to_physical("cpu", env) == ""


def test_without_mask_indices_are_physical():
    assert to_physical("0,1", {}) == "0,1"


def test_index_outside_mask_fails():
    with pytest.raises(ValueError):
        to_physical("1", {"CUDA_VISIBLE_DEVICES": "2"})


def test_docker_never_sees_all_host_gpus_under_a_mask():
    assert docker_gpu_flags("all", {"CUDA_VISIBLE_DEVICES": "2"}) == [
        "--gpus", "device=2",
    ]
    assert docker_gpu_flags("0", {"CUDA_VISIBLE_DEVICES": "3"}) == [
        "--gpus", "device=3",
    ]
    assert docker_gpu_flags("all", {}) == ["--gpus", "all"]
    assert docker_gpu_flags("cpu", {}) == []


def test_backend_envs_use_physical_ids(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert _gpu_env("0") == {"CUDA_VISIBLE_DEVICES": "2"}
    extra = protenix_runtime_env(
        {}, "0", nvcc_available=True, environ={"CUDA_VISIBLE_DEVICES": "2"}
    )
    assert extra["CUDA_VISIBLE_DEVICES"] == "2"

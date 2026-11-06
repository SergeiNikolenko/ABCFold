"""Disk-size handling of the backend environments: shared cache, dedupe,
strip/smoke gating and the container recipes."""

import os

import pytest

from abcfold.backend_envs import (FORBIDDEN_PACKAGES, BackendEnvError,
                                  BackendEnvSpec, container_recipe,
                                  dedupe_tree, image_commands,
                                  shared_cache_env, smoke_command)
from abcfold.settings import BACKEND_NAMES, load_settings

TORCH_CUDA_WHEELS = (
    "nvidia-nccl-cu12",
    "nvidia-nvshmem-cu12",
    "nvidia-cusparselt-cu12",
)


def spec_for(name):
    return BackendEnvSpec.from_settings(name, load_settings(environ={}).backend(name))


def test_uv_cache_sits_next_to_the_envs(tmp_path):
    """A cache on another filesystem makes uv copy every wheel into every env."""
    prefix = tmp_path / "envs" / "abcfold2-boltz"
    prefix.mkdir(parents=True)
    env = shared_cache_env(prefix)
    assert env["UV_LINK_MODE"] == "hardlink"
    assert os.path.dirname(env["UV_CACHE_DIR"]) == str(prefix.parent)
    assert os.path.isdir(env["UV_CACHE_DIR"])


def test_dedupe_links_identical_files_and_leaves_others(tmp_path):
    big = b"x" * (2 << 20)
    one = tmp_path / "a" / "components.cif"
    two = tmp_path / "b" / "components.cif"
    other = tmp_path / "b" / "different.cif"
    for path, data in ((one, big), (two, big), (other, b"y" * (2 << 20))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    reclaimed = dedupe_tree(tmp_path)

    assert reclaimed == len(big)
    assert one.stat().st_ino == two.stat().st_ino
    assert other.stat().st_ino != one.stat().st_ino
    assert two.read_bytes() == big
    # Idempotent: a second pass finds nothing left to link.
    assert dedupe_tree(tmp_path) == 0


def test_dedupe_ignores_small_files(tmp_path):
    for name in ("a.py", "b.py"):
        (tmp_path / name).write_bytes(b"import torch\n")
    assert dedupe_tree(tmp_path) == 0


@pytest.mark.parametrize("name", [n for n in BACKEND_NAMES if n != "alphafold3"])
def test_torch_cuda_wheels_are_never_stripped(name):
    """libtorch_cuda links NCCL, cuSPARSELt and nvSHMEM at load time, so
    removing them breaks `import torch` even for a single-GPU prediction."""
    spec = spec_for(name)
    if not spec.torch_backend or spec.name == "intellifold":
        return
    assert not set(spec.strip) & set(TORCH_CUDA_WHEELS), spec.strip


@pytest.mark.parametrize("name", [n for n in BACKEND_NAMES if n != "alphafold3"])
def test_smoke_proves_the_stripped_env_still_imports(name):
    """Anything stripped must be covered by an import that would fail."""
    spec = spec_for(name)
    if not spec.strip:
        return
    assert spec.smoke, f"{name} strips wheels without a smoke import"
    code = " ".join(smoke_command(spec))
    assert "import" in code


def test_smoke_checks_torch_itself():
    spec = spec_for("esmfold2")
    assert "torch" in spec.smoke
    assert smoke_command(spec)[:2] == ["python", "-c"]


def test_recipe_installs_the_same_pins_as_the_env():
    spec = spec_for("chai1")
    recipe = container_recipe(spec, kind="apptainer")
    for package in spec.pip:
        assert package in recipe
    for override in spec.overrides:
        assert override in recipe
    assert "kalign2" in recipe  # conda dependency
    assert f"python={spec.python}" in recipe
    assert FORBIDDEN_PACKAGES[0] in recipe  # removed inside the image too
    assert recipe.startswith("# chai1")
    assert "Bootstrap: docker" in recipe
    assert "set -e" in recipe
    assert "export PATH=/opt/abcfold/bin:$PATH" in recipe


def test_docker_recipe_is_a_dockerfile():
    recipe = container_recipe(spec_for("opendde"), kind="docker")
    assert recipe.splitlines()[1].startswith("FROM ")
    assert "RUN micromamba create" in recipe
    assert "opendde==1.2.0" in recipe


def test_keep_unused_recipe_keeps_the_wheels():
    spec = spec_for("intellifold")
    assert spec.strip
    stripped = container_recipe(spec, kind="docker", strip=True)
    kept = container_recipe(spec, kind="docker", strip=False)
    assert spec.strip[0] in stripped
    assert spec.strip[0] not in kept


def test_unknown_recipe_type_is_rejected():
    with pytest.raises(BackendEnvError):
        container_recipe(spec_for("boltz"), kind="podman-but-not-really")


def test_image_commands(tmp_path):
    spec = spec_for("boltz")
    recipe = tmp_path / "boltz.def"
    image = tmp_path / "boltz.sif"
    assert image_commands(spec, recipe, image) == [
        ["apptainer", "build", str(image), str(recipe)]
    ]
    docker = image_commands(spec, recipe, image, kind="docker")[0]
    assert docker[:2] == ["docker", "build"]
    assert "abcfold-boltz:2.2.1" in docker

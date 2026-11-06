import logging

from abcfold.backend_envs import MicromambaEnv, ensure_backend_env

logger = logging.getLogger("logger")

ENGINE_BUILD_HINT = (
    "IntelliFold needs the vendored AlphaFold 3 JAX engine, which is not "
    "distributed on PyPI. Build it into the {env} environment with:\n"
    "  git clone https://github.com/IntelliGen-AI/IntelliFold\n"
    "  cd IntelliFold\n"
    "  pip install ./third_party/alphafold3 && build_data\n"
    "Building the engine needs a C++ toolchain (cmake/ninja/compiler) and "
    "CUDA, and pulls jax[cuda12]==0.9.1."
)


def ensure_intellifold_env(config: dict) -> MicromambaEnv:
    """Verify (never modify) the pinned IntelliFold environment.

    The ``intellifold`` wheel is a thin wrapper around a vendored AlphaFold 3
    JAX engine that is not on PyPI, so a missing engine is reported with the
    build instructions instead of failing mid-run.
    """
    env = ensure_backend_env("intellifold", config)

    if env.env_exists() and env.get_installed_version("alphafold3") is None:
        raise RuntimeError(ENGINE_BUILD_HINT.format(env=env.env_name))

    return env

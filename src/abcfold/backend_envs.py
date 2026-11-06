"""Isolated backend environments.

Environments are created and populated only by ``abcfold envs build``. During a
prediction ABCFold only *verifies* that the environment exists and carries the
pinned backend version; it never upgrades or downgrades packages in place.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

from abcfold.gpus import docker_gpu_flags

logger = logging.getLogger("logger")

CURRENT_ENV = "@current"
FORBIDDEN_PACKAGES = ("rdkit-pypi",)


class BackendEnvError(RuntimeError):
    """Raised when a backend environment is missing or does not match its pin."""


def supports_cuda_wheels() -> bool:
    return sys.platform.startswith("linux")


def _clean_env(extra: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ)
    # Never leak an LD_PRELOAD shim (e.g. proxychains) into GPU workloads
    env.pop("LD_PRELOAD", None)
    if extra:
        env.update({k: str(v) for k, v in extra.items() if v is not None})
    return env


def shared_cache_env(prefix: Union[str, Path]) -> Dict[str, str]:
    """Put uv's wheel cache next to the environments so it can hardlink.

    uv hardlinks a wheel's files from its cache into the environment, but a
    hardlink cannot cross a filesystem boundary: with the default cache in
    ``~/.cache/uv`` and the environments on another mount, uv silently falls
    back to copying, and every backend gets its own 4 GB of CUDA wheels. A
    cache beside the environments makes the backends share those inodes, which
    is where most of the disk saving comes from.
    """
    cache = Path(prefix).parent / ".uv-cache"
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.debug("Could not create the shared uv cache %s: %s", cache, exc)
        return {}
    return {"UV_CACHE_DIR": str(cache), "UV_LINK_MODE": "hardlink"}


class MicromambaEnv:
    """A micromamba environment addressed by name, by prefix path, or the
    current interpreter (``@current``, e.g. inside a backend container)."""

    def __init__(self, env_name: str):
        self.env_name = env_name
        self.is_current = env_name == CURRENT_ENV
        self.is_prefix = not self.is_current and (
            os.sep in env_name or env_name.startswith("~")
        )
        self.micromamba = None if self.is_current else self._find_micromamba()

    @staticmethod
    def _find_micromamba() -> str:
        for candidate in [
            os.environ.get("MAMBA_EXE"),
            shutil.which("micromamba"),
            os.path.expanduser("~/micromamba/bin/micromamba"),
            os.path.expanduser("~/.local/bin/micromamba"),
        ]:
            if candidate and os.path.exists(candidate):
                return candidate
        raise BackendEnvError(
            "micromamba binary not found. Install micromamba or set the backend "
            "env to '@current' when running inside a backend image."
        )

    def _target(self) -> List[str]:
        if self.is_prefix:
            return ["-p", os.path.expanduser(self.env_name)]
        return ["-n", self.env_name]

    def prefix_command(self, command: List[str]) -> List[str]:
        if self.is_current:
            if command and command[0] == "python":
                return [sys.executable, *command[1:]]
            return list(command)
        assert self.micromamba is not None
        return [self.micromamba, "run", *self._target(), *command]

    def get_installed_version(self, package: str) -> Optional[str]:
        code = (
            "from importlib.metadata import version, PackageNotFoundError\n"
            "import sys\n"
            f"pkg={package!r}\n"
            "try:\n"
            "    print(version(pkg))\n"
            "except PackageNotFoundError:\n"
            "    sys.exit(3)"
        )
        try:
            out = subprocess.check_output(
                self.prefix_command(["python", "-c", code]),
                text=True,
                env=_clean_env(),
                stderr=subprocess.DEVNULL,
            )
            return out.strip() or None
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def python_prefix(self) -> Optional[str]:
        try:
            out = subprocess.check_output(
                self.prefix_command(
                    ["python", "-c", "import sys; print(sys.prefix)"]
                ),
                text=True,
                env=_clean_env(),
                stderr=subprocess.DEVNULL,
            )
            return out.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def env_exists(self) -> bool:
        if self.is_current:
            return True
        if self.is_prefix:
            return Path(os.path.expanduser(self.env_name), "bin", "python").exists()
        assert self.micromamba is not None
        try:
            out = subprocess.check_output(
                [self.micromamba, "env", "list", "--json"], text=True,
                env=_clean_env(),
            )
            prefixes = json.loads(out).get("envs", [])
            return any(Path(p).name == self.env_name for p in prefixes)
        except (subprocess.CalledProcessError, ValueError):
            return False

    def has_executable(self, name: str) -> bool:
        code = f"import shutil,sys; sys.exit(0 if shutil.which({name!r}) else 1)"
        try:
            subprocess.check_call(
                self.prefix_command(["python", "-c", code]),
                env=_clean_env(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def run(
        self,
        command: List[str],
        capture_output: bool = False,
        quiet: bool = False,
        extra_env: Optional[Mapping[str, str]] = None,
        log_file: Optional[Path] = None,
    ) -> Optional[str]:
        """Run a command inside the env.

        stdout is streamed (and returned when ``capture_output``); stderr is
        captured so that a failure carries the backend's error message.
        """
        cmd = self.prefix_command(command)
        env = _clean_env(extra_env)
        stdout_lines: List[str] = []
        stderr_lines: List[str] = []

        def pump(stream, sink, buffer):
            for line in stream:
                buffer.append(line)
                if not quiet:
                    sink.write(line)
                    sink.flush()

        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            env=env,
        ) as proc:
            assert proc.stdout is not None and proc.stderr is not None
            err_thread = threading.Thread(
                target=pump, args=(proc.stderr, sys.stderr, stderr_lines),
                daemon=True,
            )
            err_thread.start()
            pump(proc.stdout, sys.stdout, stdout_lines)
            proc.wait()
            err_thread.join()
        stdout, stderr = "".join(stdout_lines), "".join(stderr_lines)
        if log_file is not None:
            Path(log_file).write_text(stdout + "\n" + stderr)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode, cmd, output=stdout, stderr=stderr
            )
        return stdout if capture_output else None


class ContainerEnv:
    """A backend that runs inside a container image instead of its own env.

    One image replaces a per-host micromamba environment: the CUDA runtime and
    the backend are stored once, read-only, and several backends built on the
    same base share those layers. The interface matches :class:`MicromambaEnv`
    so the runners do not care which one they were given.
    """

    def __init__(self, image: str, runner: Optional[str] = None,
                 binds: Optional[List[str]] = None):
        self.image = os.path.expanduser(image)
        self.env_name = self.image
        self.is_current = False
        self.is_prefix = False
        self.runner = runner or self._detect_runner()
        self.binds = list(binds or [])
        self.installed_version: Optional[str] = None

    @staticmethod
    def _detect_runner() -> str:
        for candidate in ("apptainer", "singularity", "docker"):
            if shutil.which(candidate):
                return candidate
        raise BackendEnvError(
            "No container runtime found. Install apptainer or docker, or point "
            "the backend at a micromamba env instead of an image."
        )

    def bind_paths(self) -> List[str]:
        """Directories the backend must see: the data it reads and writes."""
        paths = [os.getcwd(), *self.binds]
        seen, out = set(), []
        for path in paths:
            resolved = str(Path(os.path.expanduser(path)).resolve())
            if resolved not in seen and Path(resolved).exists():
                seen.add(resolved)
                out.append(resolved)
        return out

    def prefix_command(self, command: List[str]) -> List[str]:
        binds = self.bind_paths()
        if self.runner == "docker":
            mounts = [arg for path in binds for arg in ("-v", f"{path}:{path}")]
            return [
                self.runner, "run", "--rm", *docker_gpu_flags(None),
                *mounts, "-w", os.getcwd(),
                # Files the backend writes must belong to the caller
                "-u", f"{os.getuid()}:{os.getgid()}",
                self.image, *command,
            ]
        mounts = [arg for path in binds for arg in ("-B", path)]
        return [
            self.runner, "exec", "--nv", "--cleanenv", *mounts,
            "--pwd", os.getcwd(), self.image, *command,
        ]

    def env_exists(self) -> bool:
        if self.runner == "docker":
            try:
                subprocess.check_call(
                    [self.runner, "image", "inspect", self.image],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    env=_clean_env(),
                )
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False
        return Path(self.image).exists()

    def get_installed_version(self, package: str) -> Optional[str]:
        return MicromambaEnv.get_installed_version(
            self, package  # type: ignore[arg-type]
        )

    def python_prefix(self) -> Optional[str]:
        return MicromambaEnv.python_prefix(self)  # type: ignore[arg-type]

    def has_executable(self, name: str) -> bool:
        return MicromambaEnv.has_executable(self, name)  # type: ignore[arg-type]

    def run(self, *args, **kwargs):
        return MicromambaEnv.run(self, *args, **kwargs)  # type: ignore[arg-type]


def pull_image(spec: "BackendEnvSpec", dry_run: bool = False) -> List[List[str]]:
    """Fetch a backend image. Apptainer builds a local .sif, docker pulls."""
    assert spec.image is not None
    env = ContainerEnv(spec.image, spec.container_runner)
    if env.runner == "docker":
        cmds = [[env.runner, "pull", spec.image]]
    else:
        source = spec.image
        target = spec.image
        if source.endswith(".sif"):
            raise BackendEnvError(
                f"{spec.name} points at the local image {source}, which does "
                f"not exist. Build it from the pins with `abcfold envs image "
                f"{spec.name} --build`, or set [backends.{spec.name}] "
                f"image = \"docker://...\" to pull a published one."
            )
        cmds = [[env.runner, "pull", target, source]]
    if not dry_run:
        for cmd in cmds:
            subprocess.check_call(cmd, env=_clean_env())
    return cmds


CONTAINER_BASE = "mambaorg/micromamba:2.3.2"


def _recipe_steps(spec: "BackendEnvSpec", strip: bool = True) -> List[str]:
    """Shell lines that install the pinned backend into /opt/abcfold."""
    remove = list(FORBIDDEN_PACKAGES) + (list(spec.strip) if strip else [])
    conda = ["python=" + spec.python, *spec.conda]
    steps = [
        "micromamba create -y -p /opt/abcfold -c conda-forge -c bioconda "
        + " ".join(conda),
        "/opt/abcfold/bin/python -m pip install --no-cache-dir uv",
    ]
    # --no-cache: a wheel cache inside the image would double the CUDA wheels
    # (about 4 GB) in the layer that creates them.
    install = ("/opt/abcfold/bin/uv pip install --no-cache "
               "--python /opt/abcfold/bin/python")
    if spec.torch_backend:
        install += f" --torch-backend {spec.torch_backend}"
    if spec.overrides:
        steps.append(
            "printf '%s\\n' " + " ".join(f"'{o}'" for o in spec.overrides)
            + " > /tmp/overrides.txt"
        )
        install += " --override /tmp/overrides.txt"
    # Quoted: extras like opendde[gpu] are glob syntax to the shell.
    steps.append(install + " " + " ".join(f"'{p}'" for p in spec.pip))
    # Tolerant: uninstalling a package that was never pulled in is not an
    # error, but the recipe runs under `set -e`.
    steps.append(
        "/opt/abcfold/bin/uv pip uninstall --python /opt/abcfold/bin/python "
        + " ".join(remove) + " || true"
    )
    steps.append("/opt/abcfold/bin/python -c \""
                 + "; ".join(f"import {m}" for m in
                             (spec.smoke or [spec.package.replace('-', '_')]))
                 + "\"")
    steps.append("micromamba clean -y --all")
    return steps


def container_recipe(spec: "BackendEnvSpec", kind: str = "apptainer",
                     strip: bool = True) -> str:
    """A Dockerfile or Apptainer definition for a backend, from the same pins.

    The image installs exactly what `abcfold envs build` installs, so a run in a
    container and a run in a local environment execute the same versions. The
    backend is on PATH, so ABCFold reaches it with
    ``[backends.<name>] image = "<image>"`` and ``env = "@current"``.
    """
    steps = _recipe_steps(spec, strip=strip)
    if kind == "docker":
        lines = [
            f"# {spec.name}: {spec.package}=={spec.version}",
            f"FROM {CONTAINER_BASE}",
            "USER root",
            "ENV MAMBA_ROOT_PREFIX=/opt/micromamba PATH=/opt/abcfold/bin:$PATH",
        ]
        lines += [f"RUN {step}" for step in steps]
        if spec.name == "intellifold":
            lines.append(
                "# The vendored AlphaFold 3 engine is not on PyPI: add\n"
                "# `COPY third_party/alphafold3 /tmp/af3` and\n"
                "# `RUN /opt/abcfold/bin/python -m pip install /tmp/af3 && "
                "build_data` from an IntelliFold checkout."
            )
        return "\n".join(lines) + "\n"
    if kind != "apptainer":
        raise BackendEnvError(f"Unknown recipe type '{kind}'")
    lines = [
        f"# {spec.name}: {spec.package}=={spec.version}",
        "Bootstrap: docker",
        f"From: {CONTAINER_BASE}",
        "",
        "%post",
        "    set -e",
        "    export MAMBA_ROOT_PREFIX=/opt/micromamba",
    ]
    lines += [f"    {step}" for step in steps]
    if spec.name == "intellifold":
        lines += [
            "    # The vendored AlphaFold 3 engine is not on PyPI: add a "
            "%files section",
            "    # copying third_party/alphafold3, pip install it here and "
            "run build_data.",
        ]
    lines += [
        "",
        "%environment",
        "    export PATH=/opt/abcfold/bin:$PATH",
        "",
        "%runscript",
        '    exec "$@"',
    ]
    return "\n".join(lines) + "\n"


def image_commands(spec: "BackendEnvSpec", recipe: Path,
                   output: Path, kind: str = "apptainer") -> List[List[str]]:
    """Command that turns a written recipe into an image."""
    if kind == "docker":
        return [["docker", "build", "-f", str(recipe), "-t",
                 spec.image or f"abcfold-{spec.name}:{spec.version}", "."]]
    return [["apptainer", "build", str(output), str(recipe)]]


@dataclass
class BackendEnvSpec:
    name: str
    package: str
    version: str
    env: str
    python: str = "3.11"
    pip: List[str] = field(default_factory=list)
    conda: List[str] = field(default_factory=list)
    overrides: List[str] = field(default_factory=list)
    torch_backend: Optional[str] = None
    # Wheels a backend's own code never references, removed after the install.
    # Each is a CUDA library torch or JAX pulls in for multi-GPU or multi-node
    # work; `smoke` is what proves the env still works without them.
    strip: List[str] = field(default_factory=list)
    smoke: List[str] = field(default_factory=list)
    image: Optional[str] = None
    container_runner: Optional[str] = None

    @classmethod
    def from_settings(cls, name: str, pins: Mapping) -> "BackendEnvSpec":
        return cls(
            name=name,
            package=pins["package"],
            version=str(pins["version"]),
            env=pins.get("env", f"abcfold2-{name}"),
            python=str(pins.get("python", "3.11")),
            pip=list(pins.get("pip", [])),
            conda=list(pins.get("conda", [])),
            overrides=list(pins.get("overrides", [])),
            torch_backend=pins.get("torch_backend"),
            strip=list(pins.get("strip", [])),
            smoke=list(pins.get("smoke", [])),
            image=pins.get("image"),
            container_runner=pins.get("container_runner"),
        )

    @property
    def uses_container(self) -> bool:
        return bool(self.image) and self.name != "alphafold3"


def smoke_command(spec: BackendEnvSpec) -> List[str]:
    """Import check that a stripped env can still load the backend."""
    modules = spec.smoke or [spec.package.replace("-", "_")]
    code = "; ".join(f"import {module}" for module in modules)
    return ["python", "-c", code + "; print('smoke ok')"]


def build_commands(
    spec: BackendEnvSpec,
    micromamba: str = "micromamba",
    uv: Optional[str] = "uv",
    python_path: Optional[str] = None,
    overrides_file: Optional[str] = None,
    strip: bool = True,
) -> List[List[str]]:
    """Commands that create and populate a backend env (pure, for testing).

    The last two commands remove the packages the backend never references and
    then import it, so a run never discovers a broken env mid-prediction.
    """
    target = ["-p", spec.env] if os.sep in spec.env else ["-n", spec.env]
    channels = ["-c", "conda-forge", "-c", "bioconda"]
    cmds = [[micromamba, "create", "-y", *target, *channels, f"python={spec.python}",
             *spec.conda]]
    py = python_path or f"<{spec.env}>/bin/python"
    remove = list(FORBIDDEN_PACKAGES) + (list(spec.strip) if strip else [])
    if uv:
        install = [uv, "pip", "install", "--python", py]
        if spec.torch_backend:
            install += ["--torch-backend", spec.torch_backend]
        if spec.overrides:
            install += ["--override", overrides_file or "<overrides.txt>"]
        cmds.append([*install, *spec.pip])
        cmds.append([uv, "pip", "uninstall", "--python", py, *remove])
    else:
        install = [micromamba, "run", *target, "python", "-m", "pip", "install"]
        if spec.torch_backend:
            install += [
                "--extra-index-url",
                f"https://download.pytorch.org/whl/{spec.torch_backend}",
            ]
        cmds.append([*install, *spec.pip, *spec.overrides])
        cmds.append([micromamba, "run", *target, "python", "-m", "pip",
                     "uninstall", "-y", *remove])
    cmds.append([micromamba, "run", *target, *smoke_command(spec)])
    return cmds


def dedupe_tree(root: Path, min_size: int = 1 << 20) -> int:
    """Hardlink byte-identical files under ``root``; return the bytes reclaimed.

    Wheels ship the same large data file more than once (the alphafold3 wheel
    installs the 525 MB libcifpp components.cif under both share/ and var/),
    and nothing writes into an env after it is built, so sharing the inodes is
    safe and survives a rebuild (uv replaces files rather than editing them).
    """
    import hashlib

    by_size: Dict[int, List[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_size < min_size:
            continue
        by_size.setdefault(stat.st_size, []).append(path)

    reclaimed = 0
    for size, paths in by_size.items():
        if len(paths) < 2:
            continue
        digests: Dict[str, Path] = {}
        for path in paths:
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                continue
            keeper = digests.get(digest)
            if keeper is None:
                digests[digest] = path
                continue
            try:
                if path.stat().st_ino == keeper.stat().st_ino:
                    continue
                temp = path.with_name(path.name + ".abcfold-dedupe")
                os.link(keeper, temp)
                os.replace(temp, path)
                reclaimed += size
            except OSError as exc:
                logger.debug("Could not link %s to %s: %s", path, keeper, exc)
    return reclaimed


def build_env(
    spec: BackendEnvSpec, dry_run: bool = False, strip: bool = True
) -> List[List[str]]:
    """Create the env and install the pinned packages. Refuses to touch an
    existing env whose backend version differs from the pin."""
    env = MicromambaEnv(spec.env)
    if env.is_current:
        raise BackendEnvError("Cannot build the '@current' environment")
    uv = shutil.which("uv")
    if env.env_exists():
        installed = env.get_installed_version(spec.package)
        if installed == spec.version:
            logger.info("%s env %s already has %s==%s", spec.name, spec.env,
                        spec.package, installed)
            return []
        if installed is not None:
            raise BackendEnvError(
                f"Env {spec.env} has {spec.package}=={installed}, pin is "
                f"{spec.version}. ABCFold does not upgrade existing envs; build a "
                f"new env (set [backends.{spec.name}] env = ...) instead."
            )
    with tempfile.TemporaryDirectory() as tmp:
        overrides_file = None
        if spec.overrides:
            overrides_file = str(Path(tmp, "overrides.txt"))
            Path(overrides_file).write_text("\n".join(spec.overrides) + "\n")
        assert env.micromamba is not None
        cmds = build_commands(spec, micromamba=env.micromamba, uv=uv,
                              overrides_file=overrides_file, strip=strip)
        if dry_run:
            return cmds
        create, *_rest = cmds
        if not env.env_exists():
            subprocess.check_call(create, env=_clean_env())
        prefix = env.python_prefix()
        if prefix is None:
            raise BackendEnvError(f"Could not locate the python of env {spec.env}")
        cmds = build_commands(spec, micromamba=env.micromamba, uv=uv,
                              python_path=str(Path(prefix, "bin", "python")),
                              overrides_file=overrides_file, strip=strip)
        install, uninstall, smoke = cmds[1], cmds[2], cmds[3]
        uv_env = shared_cache_env(prefix) if uv else {}
        subprocess.check_call(install, env=_clean_env(uv_env))
        subprocess.call(uninstall, env=_clean_env(uv_env))
        try:
            subprocess.check_call(smoke, env=_clean_env())
        except subprocess.CalledProcessError as exc:
            raise BackendEnvError(
                f"{spec.name} does not import in env '{spec.env}' after the "
                f"install. If the packages removed as unused are the cause, "
                f"rebuild with `abcfold envs build {spec.name} --keep-unused`."
            ) from exc
        # Deduplicate across all sibling environments: a wheel uv had to copy
        # (older envs, or a cache on another filesystem) collapses back onto a
        # single inode here.
        scope = Path(prefix).parent if Path(prefix).parent.name == "envs" \
            else Path(prefix)
        reclaimed = dedupe_tree(scope)
        if reclaimed:
            logger.info("Deduplicated %.0f MB in %s", reclaimed / 1024 / 1024,
                        scope)
    return cmds


def verify_env(spec: BackendEnvSpec, allow_mismatch: bool = False) -> str:
    """Check (never modify) a backend env. Returns the installed version."""
    env = MicromambaEnv(spec.env)
    if not env.env_exists():
        raise BackendEnvError(
            f"{spec.name} env '{spec.env}' does not exist. Build it with "
            f"`abcfold envs build {spec.name}`."
        )
    installed = env.get_installed_version(spec.package)
    if installed is None:
        raise BackendEnvError(
            f"{spec.package} is not installed in env '{spec.env}'. Build it with "
            f"`abcfold envs build {spec.name}` (or point the backend at another env)."
        )
    if installed != spec.version:
        msg = (
            f"{spec.package}=={installed} in env '{spec.env}' does not match the "
            f"pinned version {spec.version}"
        )
        if not allow_mismatch:
            raise BackendEnvError(
                msg + ". ABCFold does not upgrade envs at run time; build a new env "
                "or override the pin in your config file."
            )
        logger.warning(msg)
    for forbidden in FORBIDDEN_PACKAGES:
        if env.get_installed_version(forbidden) is not None:
            raise BackendEnvError(
                f"Env '{spec.env}' contains {forbidden}, which shadows rdkit. "
                f"Remove it: micromamba run -n {spec.env} pip uninstall -y {forbidden}"
            )
    return installed


def container_binds(settings) -> List[str]:
    """Weight and cache directories a containerised backend must see."""
    keys = (
        "weights_dir", "boltz_weights", "openfold_weights",
        "rosettafold_weights", "intellifold_weights", "protenix_root_dir",
        "opendde_root_dir", "esmfold2_weights", "chai_downloads_dir",
        "msa_cache_dir", "template_mirror_dir",
    )
    binds = []
    for key in keys:
        value = settings.get(key)
        if value:
            binds.append(str(value))
    return binds


def ensure_backend_env(
    name: str, config: Mapping
) -> Union[MicromambaEnv, ContainerEnv]:
    """Verify the env of ``name`` using the runtime config and return it."""
    settings = config.get("_settings")
    if settings is None:
        from abcfold.settings import load_settings

        settings = load_settings()
    spec = BackendEnvSpec.from_settings(name, settings.backend(name))
    if spec.uses_container:
        container = ContainerEnv(
            spec.image or "", spec.container_runner, container_binds(settings)
        )
        if not container.env_exists():
            raise BackendEnvError(
                f"{name} image '{container.image}' is not available. Fetch it "
                f"with `abcfold envs build {name}`."
            )
        installed = container.get_installed_version(spec.package)
        if installed is not None and installed != spec.version and not bool(
            config.get("allow_version_mismatch", False)
        ):
            raise BackendEnvError(
                f"{spec.package}=={installed} in image '{container.image}' does "
                f"not match the pinned version {spec.version}"
            )
        container.installed_version = installed or spec.version
        logger.info("%s %s (image %s)", name, container.installed_version,
                    container.image)
        return container
    env = MicromambaEnv(spec.env)
    if (
        not env.is_current
        and not env.env_exists()
        and settings.get("build_missing_envs", False)
    ):
        logger.info("Building missing %s env %s", name, spec.env)
        build_env(spec)
    allow = bool(config.get("allow_version_mismatch", False))
    if env.is_current:
        installed = env.get_installed_version(spec.package) or "unknown"
    else:
        installed = verify_env(spec, allow_mismatch=allow)
    env.installed_version = installed  # type: ignore[attr-defined]
    logger.info("%s %s (env %s)", name, installed, spec.env)
    return env

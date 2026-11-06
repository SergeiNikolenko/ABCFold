"""Read-only configuration for ABCFold.

Precedence (lowest to highest):

1. packaged ``data/defaults.toml`` and ``backends.toml`` (read-only),
2. a user config file given by ``--config-file`` or ``ABCFOLD_CONFIG``
   (TOML, or a legacy ABCFold ``config.ini``); it is read, never written,
3. ``ABCFOLD_<KEY>`` environment variables,
4. command line flags.

Unset values are ``None``; the strings ``""`` and ``"None"`` count as unset.
"""

from __future__ import annotations

import configparser
import copy
import os
import sys
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

ENV_PREFIX = "ABCFOLD_"
BACKEND_NAMES = (
    "boltz",
    "chai1",
    "protenix",
    "openfold3",
    "rosettafold3",
    "intellifold",
    "opendde",
    "esmfold2",
    "alphafold3",
)

# Keys used by the legacy flat runtime config (config.ini era)
LEGACY_ENV_KEYS = {
    "boltz_env": "boltz",
    "chai_env": "chai1",
    "protenix_env": "protenix",
    "openfold_env": "openfold3",
    "rosettafold_env": "rosettafold3",
    "intellifold_env": "intellifold",
    "opendde_env": "opendde",
    "esmfold2_env": "esmfold2",
}
LEGACY_VERSION_KEYS = {
    "boltz_version": "boltz",
    "chai_version": "chai1",
    "protenix_version": "protenix",
    "openfold_version": "openfold3",
    "rosetta_version": "rosettafold3",
    "intellifold_version": "intellifold",
    "opendde_version": "opendde",
    "esmfold2_version": "esmfold2",
    "af3_version": "alphafold3",
}


def _unset(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() in {"", "None"})


def normalize_value(value: Any) -> Any:
    """Map the unset markers to None and leave other values untouched."""
    return None if _unset(value) else value


def _read_toml_resource(name: str) -> dict:
    text = resources.files("abcfold").joinpath(name).read_text()
    return tomllib.loads(text)


def load_packaged_defaults() -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for section in _read_toml_resource("data/defaults.toml").values():
        if isinstance(section, dict):
            for key, value in section.items():
                flat[key] = normalize_value(value)
    return flat


def load_backend_pins() -> Dict[str, Dict[str, Any]]:
    data = _read_toml_resource("backends.toml")
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def _read_user_file(path: Path) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Return (flat values, backend overrides) from a TOML or legacy INI file."""
    values: Dict[str, Any] = {}
    backends: Dict[str, Dict[str, Any]] = {}
    if path.suffix == ".toml":
        data = tomllib.loads(path.read_text())
        for key, section in data.items():
            if key == "backends" and isinstance(section, dict):
                for name, override in section.items():
                    backends[name] = dict(override)
            elif isinstance(section, dict):
                for sub_key, value in section.items():
                    values[sub_key] = normalize_value(value)
            else:
                values[key] = normalize_value(section)
        return values, backends

    parser = configparser.ConfigParser()
    parser.read(path)
    for section in parser.sections():
        for key, value in parser.items(section):
            value = normalize_value(value)
            if key in LEGACY_VERSION_KEYS:
                if value is not None:
                    backends.setdefault(LEGACY_VERSION_KEYS[key], {})["version"] = value
            elif key in LEGACY_ENV_KEYS:
                if value is not None:
                    backends.setdefault(LEGACY_ENV_KEYS[key], {})["env"] = value
            elif key == "protenix_model":
                if value is not None:
                    backends.setdefault("protenix", {})["model"] = value
            else:
                values[key] = value
    return values, backends


def _coerce_like(value: str, reference: Any) -> Any:
    if isinstance(reference, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(reference, int):
        try:
            return int(value)
        except ValueError:
            return value
    return value


@dataclass
class Settings:
    values: Dict[str, Any] = field(default_factory=dict)
    backends: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    source_file: Optional[Path] = None

    def get(self, key: str, default: Any = None) -> Any:
        value = self.values.get(key)
        return default if value is None else value

    def path(self, key: str) -> Optional[Path]:
        value = self.get(key)
        return None if value is None else Path(str(value)).expanduser()

    def backend(self, name: str) -> Dict[str, Any]:
        if name not in self.backends:
            raise KeyError(f"Unknown backend '{name}'")
        return self.backends[name]

    def apply_overrides(self, overrides: Mapping[str, Any]) -> "Settings":
        """Return a copy with non-None overrides applied (e.g. CLI flags)."""
        new = copy.deepcopy(self)
        for key, value in overrides.items():
            if value is not None:
                new.values[key] = value
        return new

    def runtime_config(self) -> Dict[str, Any]:
        """Flat dict with the keys the backend runners historically used."""
        cfg: Dict[str, Any] = dict(self.values)
        for key, name in LEGACY_ENV_KEYS.items():
            cfg[key] = self.backends.get(name, {}).get("env")
        for key, name in LEGACY_VERSION_KEYS.items():
            cfg[key] = self.backends.get(name, {}).get("version")
        cfg["protenix_model"] = self.backends.get("protenix", {}).get("model")
        cfg["openfold_checkpoint"] = self.backends.get("openfold3", {}).get(
            "checkpoint"
        )
        cfg["opendde_model"] = self.backends.get("opendde", {}).get("model")
        cfg["esmfold2_model"] = self.backends.get("esmfold2", {}).get("model")
        cfg["esmfold2_device"] = self.backends.get("esmfold2", {}).get("device")
        cfg["_settings"] = self
        return cfg


def load_settings(
    config_file: Optional[os.PathLike | str] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Settings:
    """Load settings without writing anything to disk."""
    environ = os.environ if environ is None else environ
    values = load_packaged_defaults()
    backends = load_backend_pins()

    if config_file is None:
        config_file = environ.get(ENV_PREFIX + "CONFIG") or None
    source = None
    if config_file is not None:
        source = Path(config_file).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"Config file not found: {source}")
        user_values, user_backends = _read_user_file(source)
        values.update({k: v for k, v in user_values.items()})
        for name, override in user_backends.items():
            backends.setdefault(name, {}).update(override)
            # A version override without a pip override must still be honoured
            if "version" in override and "pip" not in override:
                package = backends[name].get("package")
                pins = backends[name].get("pip", [])
                backends[name]["pip"] = [
                    _replace_pin(spec, package, override["version"]) for spec in pins
                ]

    for key in list(values):
        env_key = ENV_PREFIX + key.upper()
        if env_key in environ:
            values[key] = normalize_value(
                _coerce_like(environ[env_key], values.get(key))
            )
    for name in BACKEND_NAMES:
        for field_name in ("version", "env"):
            env_key = f"{ENV_PREFIX}{name.upper()}_{field_name.upper()}"
            if env_key in environ and name in backends:
                backends[name][field_name] = environ[env_key]
                if field_name == "version":
                    package = backends[name].get("package")
                    backends[name]["pip"] = [
                        _replace_pin(spec, package, environ[env_key])
                        for spec in backends[name].get("pip", [])
                    ]

    return Settings(values=values, backends=backends, source_file=source)


def _replace_pin(spec: str, package: Optional[str], version: str) -> str:
    """Replace the ``==`` version of ``package`` in a pip requirement string."""
    if package is None or "==" not in spec:
        return spec
    name, _ = spec.split("==", 1)
    base = name.split("[", 1)[0].strip().lower().replace("-", "_")
    if base == package.lower().replace("-", "_"):
        return f"{name}=={version}"
    return spec

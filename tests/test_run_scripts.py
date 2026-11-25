import subprocess
import sys


def _run_help(module: str) -> bytes:
    proc = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"Return code: {proc.returncode}\n{proc.stderr.decode()}"
    )
    return proc.stdout


def test_alphafold3_script():
    stdout = _run_help("abcfold.abcfold")
    assert b"usage:" in stdout and b"abcfold" in stdout


def test_add_custom_template_script():
    stdout = _run_help("abcfold.scripts.add_custom_template")
    assert b"usage:" in stdout and b"add_custom_template" in stdout


def test_add_mmseqs_msa_script():
    stdout = _run_help("abcfold.scripts.add_mmseqs_msa")
    assert b"usage:" in stdout and b"add_mmseqs_msa" in stdout


def test_ipsae_script():
    stdout = _run_help("abcfold.scripts.ipsae")
    assert b"usage:" in stdout and b"ipsae" in stdout


def test_boltz_affinity_script():
    stdout = _run_help("abcfold.scripts.boltz_affinity")
    assert b"usage:" in stdout and b"boltz_affinity" in stdout


def test_dry_run_script():
    stdout = _run_help("abcfold.dry_run")
    assert b"usage:" in stdout

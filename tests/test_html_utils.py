from pathlib import Path

from abcfold.backend_envs import MicromambaEnv
from abcfold.html.html_utils import get_model_data, plots


def test_plots_can_skip_pae_for_ccp4cloud(output_objs, tmp_path):
    plot_dict = plots(
        [output_objs.boltz_output], tmp_path, include_pae=False
    )
    assert "plddt" in plot_dict
    assert Path(plot_dict["plddt"]).exists()
    # Only the pLDDT plot is written; no per-model PAE HTML files.
    assert list(plot_dict.keys()) == ["plddt"]


def test_get_model_data_includes_affinity_and_handles_missing_pae(
    output_objs, tmp_path
):
    boltz = output_objs.boltz_output
    files = boltz.output["seed-1"][0]
    model = files["cif"]
    model.check_clashes()
    out_root = model.pathway.parents[3]
    aff = {
        str(model.pathway.resolve()): {
            "affinity_pred_value": -1.234,
            "affinity_probability_binary": 0.912,
        }
    }
    data = get_model_data(
        model,
        {},
        "Boltz",
        model.residue_plddts,
        files["af3_pae"],
        files["json"],
        out_root,
        affinity_scores=aff,
    )
    assert data["pae_path"] is None
    assert data["affinity_pred_value"] == -1.234
    assert data["affinity_probability_binary"] == 0.912
    assert "reactifptm_score" in data


def test_current_env_which():
    env = MicromambaEnv("@current")
    assert env.which("python3") is True
    assert env.which("definitely_not_a_real_binary_xyz123") is False

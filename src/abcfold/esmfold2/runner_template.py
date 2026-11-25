"""Runner executed inside the isolated ESMFold2 environment.

ESMFold2 exposes no command line interface, so abcfold writes this script next
to the converted input and runs it with the environment's interpreter. It reads
either a single input description or a multi-job --manifest, loads the
EsmFold2Model once onto GPU with TF32 + inference_mode, folds every job/seed,
and writes mmCIF models plus AlphaFold3 style confidence summaries.

Usage:
    python runner.py <input_description.json> <output_dir> <options.json>
    python runner.py --manifest <manifest.json>
"""

import csv
import inspect
import json
import sys
from pathlib import Path


def build_input(description, esm_mod):
    ProteinInput = esm_mod.ProteinInput
    RNAInput = esm_mod.RNAInput
    DNAInput = esm_mod.DNAInput
    LigandInput = esm_mod.LigandInput
    Modification = esm_mod.Modification
    StructurePredictionInput = esm_mod.StructurePredictionInput

    from esm.utils.msa import MSA

    polymer_types = {"protein": ProteinInput, "rna": RNAInput, "dna": DNAInput}

    sequences = []
    for entry in description["sequences"]:
        entity_type = entry["type"]
        chain_id = entry["id"]
        if isinstance(chain_id, list) and len(chain_id) == 1:
            chain_id = chain_id[0]

        if entity_type == "ligand":
            sequences.append(
                LigandInput(
                    id=chain_id,
                    ccd=entry.get("ccd"),
                    smiles=entry.get("smiles"),
                )
            )
            continue

        kwargs = {"id": chain_id, "sequence": entry["sequence"]}

        modifications = entry.get("modifications")
        if modifications:
            kwargs["modifications"] = [
                Modification(position=mod["position"], ccd=mod["ccd"])
                for mod in modifications
            ]

        msa_path = entry.get("msa_path")
        if msa_path and entity_type in ("protein", "rna"):
            kwargs["msa"] = MSA.from_a3m(path=msa_path)

        sequences.append(polymer_types[entity_type](**kwargs))

    return StructurePredictionInput(sequences=sequences)


def normalise_plddt(plddt):
    if plddt is None:
        return None
    value = float(plddt.mean())
    return value * 100 if value <= 1.0 else value


def _configure_torch():
    import torch

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return torch


def _fold_job(builder, model, esm_mod, torch_mod, description, output_dir, options):
    name = description.get("name") or "prediction"
    structure_input = build_input(description, esm_mod)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_params = inspect.signature(builder.fold).parameters
    extra_fold_kwargs = {}
    if "mode" in fold_params:
        requested_mode = options.get("fold_mode")
        if requested_mode:
            extra_fold_kwargs["mode"] = requested_mode
        elif torch_mod.cuda.is_available():
            extra_fold_kwargs["mode"] = "fused"

    ranking_rows = []
    for seed in description["seeds"]:
        seed_dir = output_dir / "esmfold2_results_seed-{}".format(seed)
        seed_dir.mkdir(parents=True, exist_ok=True)

        with torch_mod.inference_mode():
            results = builder.fold(
                model,
                structure_input,
                num_loops=options["num_loops"],
                num_sampling_steps=options["num_sampling_steps"],
                num_diffusion_samples=options["num_diffusion_samples"],
                seed=seed,
                lm_dropout=0.0,
                complex_id=name,
                **extra_fold_kwargs,
            )
        if not isinstance(results, list):
            results = [results]

        for index, result in enumerate(results):
            stem = "{}_seed-{}_sample-{}".format(name, seed, index)
            (seed_dir / "{}_model.cif".format(stem)).write_text(
                result.complex.to_mmcif()
            )

            ptm = float(result.ptm) if result.ptm is not None else None
            iptm = float(result.iptm) if result.iptm is not None else None
            plddt = normalise_plddt(result.plddt)
            if iptm is not None and ptm is not None:
                ranking_score = 0.8 * iptm + 0.2 * ptm
            elif ptm is not None:
                ranking_score = ptm
            else:
                ranking_score = plddt / 100 if plddt is not None else 0.0

            summary = {
                "ptm": ptm,
                "iptm": iptm,
                "plddt": plddt,
                "ranking_score": ranking_score,
                "num_recycles": options["num_loops"],
            }
            (seed_dir / "{}_summary_confidences.json".format(stem)).write_text(
                json.dumps(summary, indent=4)
            )

            if result.pae is not None:
                (seed_dir / "{}_confidences.json".format(stem)).write_text(
                    json.dumps(
                        {"pae": result.pae.tolist()},
                        separators=(",", ":"),
                    )
                )

            ranking_rows.append([seed, index, ranking_score])

    with open(output_dir / "{}_ranking_scores.csv".format(name), "w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "sample", "ranking_score"])
        writer.writerows(ranking_rows)


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--manifest":
        manifest = json.loads(Path(sys.argv[2]).read_text())
        jobs = manifest["jobs"]
        base_options = manifest["options"]
    else:
        description_path, output_arg, options_path = sys.argv[1:4]
        base_options = json.loads(Path(options_path).read_text())
        jobs = [
            {
                "description": description_path,
                "output_dir": output_arg,
                "options": base_options,
            }
        ]

    torch_mod = _configure_torch()
    import esm.models.esmfold2 as esm_mod
    from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model

    builder = ESMFold2InputBuilder()
    model = EsmFold2Model.from_pretrained(
        base_options["model_repo"], device=base_options["device"]
    ).eval()

    for job in jobs:
        description = json.loads(Path(job["description"]).read_text())
        options = job.get("options") or base_options
        _fold_job(
            builder,
            model,
            esm_mod,
            torch_mod,
            description,
            Path(job["output_dir"]),
            options,
        )


if __name__ == "__main__":
    main()

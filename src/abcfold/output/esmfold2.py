from abcfold.output.cofolding_common import SeedSampleOutput


class Esmfold2Output(SeedSampleOutput):
    """
    Object to process the output of an ESMFold2 run.

    ESMFold2 returns results in memory, so abcfold's runner writes them as
    ``esmfold2_results/esmfold2_results_seed-<seed>/<job>_seed-<seed>_sample-<n>_model.cif``
    alongside ``*_summary_confidences.json`` and ``*_confidences.json``.
    ``MolecularComplex.to_mmcif`` writes pLDDT into the B-factor column already
    scaled to 0-100.
    """

    results_prefix = "esmfold2_results"
    label = "esmfold2"
    display_name = "ESMFold2"
    sample_token = "_sample-"
    plddt_is_fraction = False

    def seed_for(self, output_dir, pathway):
        # One run directory holds every seed, so the seed comes from the
        # esmfold2_results_seed-<seed> directory each model was written to.
        for part in pathway.parts[::-1]:
            if part.startswith("esmfold2_results_seed-"):
                return part.rsplit("-", 1)[-1]
        return super().seed_for(output_dir, pathway)

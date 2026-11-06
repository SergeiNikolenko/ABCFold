# ABCFold

![Build Status](https://github.com/rigdenlab/ABCFold/actions/workflows/python-package.yml/badge.svg)
![Coverage](https://raw.githubusercontent.com/rigdenlab/ABCFold/refs/heads/main/.blob/coverage.svg)


Scripts to run AlphaFold3, Boltz, Chai-1, OpenFold3, Protenix, RosettaFold3, IntelliFold, OpenDDE and ESMFold2 with MMseqs2 Multiple sequence alignments (MSAs) and custom templates.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Common Issues](#common-issues)
- [Contributing](#contributing)

## Installation

We recommend working with this repository through `uv run` in a micromamba-backed setup. Python 3.11 is recommended, but the package should work with Python 3.10 and above.
For instructions on installing micromamba, check their website [here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

To set up a micromamba environment, run:
```bash
micromamba create -n abcfold python=3.11 -y
micromamba activate abcfold
```

> **Environment note**
>
> - Your main Python environment can be Conda, micromamba, or virtualenv.
> - `abcfold` will automatically create **internal micromamba environments** to run Boltz, Chai-1, OpenFold3, Protenix, RosettaFold3, IntelliFold, OpenDDE and ESMFold2 safely, therefore a micromamba installation is required.
> - This prevents package conflicts with your main environment and ensures reproducible results.

We recommend installing and running ABCFold via `uv`:

```bash
uv sync
```

If you prefer installing the package from PyPI, run:

```bash
uv tool install abcfold
```

Or, to install the package from source into the local project environment, first clone the repository and then run:

```bash
uv sync
```

## Development

If you wish to help develop this package, install the development dependencies with `uv`:

```bash
uv sync --extra dev
```

We recommend enabling local Git hooks with Lefthook:

```bash
brew install lefthook
lefthook install
```

Typical development commands should be run through `uv run`:

```bash
uv run --extra test pytest
uv run --extra lint ruff check abcfold tests
uv run --extra lint ruff format abcfold tests
```

The provided `lefthook.yml` follows the same `uv run` workflow: staged Python files are checked on `pre-commit`, and a targeted lint plus test suite is run on `pre-push`.

## Usage

### Running ABCfold

ABCFold will run Alphafold3, Boltz, Chai-1, OpenFold3, Protenix, RosettaFold3, IntelliFold, OpenDDE and ESMFold2 consecutively. The program takes an input of a JSON in the Alphafold3 format (For full instruction on how to format this, click [here](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md)). An example JSON is shown below:

```json
{
  "name": "2PV7",
  "sequences": [
    {
      "protein": {
        "id": ["A", "B"],
        "sequence": "GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG"
      }
    }
  ],
  "modelSeeds": [1],
  "dialect": "alphafold3",
  "version": 1
}
```

By default, the AF3 backend is run through the `AlphaFast` container image. You still need access to the AlphaFold3 model parameters; the other backends run in their dedicated micromamba environments, pinned in `backends.toml` and built with `abcfold envs build`.

For the majority of jobs, ABCFold can be run as follows:
```bash
uv run abcfold <input_json> <output_dir> -abcop --mmseqs2 --model_params <path_to_af3_model_params>
```
> [!NOTE]
> `--model_params` is stored after the first run, therefore subsequent ABCFold jobs don't require this flag.

> [!NOTE]
> If you wish to run ABCFold with the AlphaFold3 JACKHMMER MSA search, you need to remove the `--mmseqs2` flag and provide the `--database` flag with the path to the directory containing the AlphaFold3 databases.
> The `--database` path will also be stored after the first run and won't be required in subsequent ABCFold jobs.

>[!WARNING]
> When using the `--mmseqs2` flag, AlphaFold3 will be run without pairedMSA information. If this is important for your target (e.g. modelling a complex), we recommend running the AlphaFold3 JACKHMMER MSA search as the pairedMSA is automatically generated.

>[!WARNING]
>`--model_params` and `--database` will need to be provided again if you do a fresh install.

However, there you may wish to use the following flags to add run time options such as the use of templates, or the number of models to create.

#### Main arguments
- `<input_json>`: Path to the input AlphaFold3 JSON file.
- `<output_dir>`: Path to the output directory.
- `-a`, `-b`, `-c`, `-o`, `-p`, `-r`, `-i`, `-d`, `-e` (`--alphafold3`, `--boltz`,`--chai1`, `--openfold3`, `--protenix`, `--rosettafold3`, `--intellifold`, `--opendde`, `--esmfold2`): Flags to run Alphafold3, Boltz, Chai-1, OpenFold3, Protenix, RosettaFold3, IntelliFold, OpenDDE and ESMFold2 respectively. If none of these flags are provided, Alphafold3 will be run by default.
- `--intellifold_weights`: [optional] Directory holding the IntelliFold weights; by default they are downloaded to `~/.intellifold/model_v2/`.

IntelliFold, OpenDDE and ESMFold2 take MSAs but no templates, so a template search is not forwarded to them. `--num_recycles` and `--gpus` reach IntelliFold and OpenDDE; ESMFold2 honours `--gpus` and keeps its own 20 trunk loops. IntelliFold always runs with its data pipeline off, so a chain without an MSA is folded from its single sequence. ESMFold2 has no command line interface: ABCFold drives its Python API from a runner script inside the pinned environment and writes the AlphaFold3 style confidence files itself, deriving the missing ranking score as `0.8 * ipTM + 0.2 * pTM`. IntelliFold additionally needs its vendored AlphaFold3 JAX engine built into the environment (`pip install ./third_party/alphafold3 && build_data` from the IntelliFold checkout); ABCFold reports this up front instead of failing mid-run.
- `--mmseqs2`: [optional] Flag to use MMseqs2 MSAs and templates (if specified).
- `--mmseqs_database`: [optional] The path to the database used by a local copy of MMSeqs2, provided mmseqs is installed, the inclusion of this flag allows MMseqs2 to be run locally.
- `--override`: [optional] Flag to override the existing output directory.
- `--save_input`: [optional] Flag to save the input JSON file in the output directory.

#### Alphafold3 arguments

- `--model_params`: Path to the directory containing the AlphaFold3 model parameters.
- `--database`: [optional] Path to the directory containing the AlphaFold3 databases #Note: This is not used if using the
`--mmseqs2` flag.
- `--af3_sif_path`: [optional] Path to sif file if using an AlphaFold3 singularity instead of Docker
- `--use_af3_template_search`[optional] If providing your own custom MSA or you've ran `--mmseqs2`, allow Alphafold3 to search for templates

#### OpenFold3 arguments
- `--inference_ckpt_path` [optional] Path for model checkpoint to be used for inference. If not specified, will attempt to find or download parameters in ~/.openfold3/

#### Chai-1 fast runner

`--chai1` uses ABCFold's persistent Chai worker by default. The worker keeps
the same Chai quality settings as the standard runner while reducing repeated
process setup overhead and distributing seed jobs across the requested
`--gpus` slots. `--chai_fast` was intentionally removed; use `--chai1`.
See [Chai Fast Runner](docs/chai_fast_runner.md) for smoke checks and details.

#### Template arguments

- `--templates`: Flag to enable a template search
- `--num_templates`: [optional] The number of templates to use (default: 20)

- `--custom_template`: [optional] Path to a custom template file in mmCIF format or a list of custom templates. A more detailed decription on how to use the custom template argument can be found below Visualisation arguments.
- `--custom_template_chain`: [conditionally required] The chain ID of the chain to use in your custom template, only required if using a multi-chain template. If providing a list of custom templates, you will need to provide a list of custom template chains.
- `--target_id`: [conditionally required] The ID of the sequence the custom template relates to, only required if modelling a complex. If providing a list of custom templates, you can provide a single target ID if they all relate to the same target. Otherwise, you should provide a list of target IDs corresponding to the list of custom templates.

#### Visualisation arguments
- `--no_server`: [optional] Flag to not run the server for the output page (see below) but the pages are created , useful for running on a cluster.
- `--no_visuals`: [optional] Flag to not generate any output pages or PAE plots and only output the models.

#### Custom template usage

If you wanted to provide a custom template, `custom_a.pdb` for your protein sequence with the ID `A` and you have your template has two chains: chain `A` and chain `B` and chain `B` is what you want the template to be, you could run:

```bash
uv run abcfold <input_json> <output_dir> -abcop --mmseqs2 --custom_template custom_a.pdb --custom_template_chain B --target_id A

```

If you had multiple IDs in your input sequence, multiple template files and you wanted to provide 3 custom templates, chain `A` from `custom_a.pdb`, chain `B` from `custom_b.pdb`, and chain B from `custom_c.pdb`, where `custom_a.pdb` and `custom_b.pdb` correspond to the ID `A` and `custom_c.pdb` corresponds to the ID `B`, you could run:

```bash
uv run abcfold <input_json> <output_dir> -abcop --mmseqs2 --custom_template custom_a.pdb custom_b.pdb custom_c.pdb --custom_template_chain A B B --target_id A A B

```
### Backend environments and containers

Each backend lives in its own micromamba environment, pinned in `backends.toml`
and built with `abcfold envs build [backend ...]`. `abcfold envs show` lists the
pins, `abcfold envs check` verifies an environment without modifying it.

A GPU backend environment is 6-7 GB, and almost all of it is PyTorch and the
CUDA wheels it depends on (about 4 GB of `nvidia-*` plus 1.5-1.8 GB of torch).
Those cannot be trimmed: `libtorch_cuda.so` links `libnccl.so.2`,
`libcusparseLt.so.0` and `libnvshmem_host.so.3` at load time, so removing
`nvidia-nccl-cu12`, `nvidia-cusparselt-cu12` or `nvidia-nvshmem-cu12` makes
`import torch` fail even for a single-GPU prediction. The JAX engine behind
IntelliFold does load its collectives lazily, so that environment drops NCCL
and nvSHMEM (about 870 MB) and still sees every GPU. Each backend's `strip`
list is guarded by a `smoke` import, so a build that breaks an environment
fails at build time instead of mid-prediction; `--keep-unused` skips stripping.

ABCFold reduces the footprint in two ways that do not touch what a backend
loads:

- The uv wheel cache is placed next to the environments during a build. uv
  hardlinks wheel files from its cache, but a hardlink cannot cross a
  filesystem boundary: with the default `~/.cache/uv` on another mount, uv
  silently copies, and every backend gets its own copy of every wheel.
- `abcfold envs dedupe` hardlinks byte-identical files across all backend
  environments, which also fixes environments built before this. On a three
  backend install it reclaims about 1 GB (the `alphafold3` wheel alone ships
  the same 525 MB `components.cif` twice).

Model weights are downloaded only for the backends you actually request, and
the `[paths]` settings keep them out of the environments: `boltz_weights`,
`openfold_weights`, `rosettafold_weights`, `intellifold_weights`,
`protenix_root_dir`, `opendde_root_dir`, `esmfold2_weights` (the Hugging Face
cache root) and `chai_downloads_dir`. Point them at shared storage and several
hosts and environments use one copy.

For a container instead of a local environment, `abcfold envs image <backend>`
writes an Apptainer definition (or `--recipe docker` for a Dockerfile) built
from the same pins, so a containerised run executes the same versions as a
local one. Add `--build` to build it immediately:

```bash
abcfold envs image opendde --output-dir containers --build
```

Then point the backend at the image in your config file:

```toml
[backends.opendde]
image = "containers/abcfold-opendde-1.2.0.sif"
```

ABCFold mounts the working directory and the configured weight and cache
directories into the container and passes the GPU through (`--nv` for
Apptainer, `--gpus` for Docker).

### Output

ABCFold will output the AlphaFold, Boltz, Chai, OpenFold3, Protenix, RosettaFold3, IntelliFold, OpenDDE and ESMFold2 models in the `<output_dir>`, it will also produce an output page containing a results table and informative [PAE viewer](https://gitlab.gwdg.de/general-microbiology/pae-viewer). This is opened automatically in your default browser unless the `--no_server` or `--no_visuals` flags are used.

Unless the `--no_visuals` flag is used, you can then open the output pages by running:

```bash
cd <output_dir>
uv run python open_output.py
```

## Main Page Example
![main_page_example](https://raw.githubusercontent.com/rigdenlab/ABCFold/refs/heads/main/abcfold/html/static/main_page_example.png)

## PAE Viewer example
![pae_viewer_example](https://raw.githubusercontent.com/rigdenlab/ABCFold/refs/heads/main/abcfold/html/static/pae_viewer_example.png)

The output page will be available on `http://localhost:8000/index.html`. If you need to rerun the server to create the output,
you will find `open_output.py` in your `<output_dir>`. This needs to be run from your `<output_dir>`.


## Extra Features

Below are scripts for adding MMseqs2 MSAs and custom templates to AlphaFold3 input JSON files.

> [!WARNING]
> These scripts will only modify the input JSON files, I.E. they will NOT run any of the prediction backends.

### Scoring existing Boltz complexes and poses

ABCFold also includes a Boltz2 utility for scoring existing complex coordinates
or fixed-receptor ligand poses without running Boltz diffusion sampling. See
[Boltz Affinity Scoring](docs/boltz_affinity.md) for
confidence scoring, affinity scoring, and `--reuse_trunk` examples.

This is the fast Boltz path for already-built poses: it loads Boltz2 and runs
the confidence and optional affinity heads on supplied coordinates, but skips
the expensive diffusion structure-generation loop.

After installation, use `boltz-affinity` directly:

```bash
boltz-affinity poses.sdf --receptor receptor.pdb --affinity
```

For a checkout managed with `uv`, either run through `uv`:

```bash
uv run boltz-affinity poses.sdf --receptor receptor.pdb --affinity
```

or activate the project environment first:

```bash
source .venv/bin/activate
boltz-affinity poses.sdf --receptor receptor.pdb --affinity
```

The module form
`python -m abcfold.boltz.score_existing` remains available for development
checkouts.

### Docking ligand SMILES into a crystal pocket with Boltz

ABCFold also includes a Boltz-native docking wrapper for the case where a
crystal receptor and pocket are known, but the ligand pose should be generated
from SMILES by Boltz. See
[Boltz Dock](docs/boltz_dock.md) for crystal
template, pocket constraint, reference-ligand, and affinity examples.

This mode is named `boltz-dock` because it generates a new ligand pose. It is
not as fast as `boltz-affinity`: Boltz diffusion still runs, but the receptor is
kept close to the provided crystal template and the ligand is steered into the
known pocket.

After installation, use `boltz-dock` directly:

```bash
boltz-dock receptor.pdb "CCO" --pocket_residue A:145 --affinity
```

For a checkout managed with `uv`, either run through `uv`:

```bash
uv run boltz-dock receptor.pdb "CCO" --pocket_residue A:145 --affinity
```

or activate the project environment first:

```bash
source .venv/bin/activate
boltz-dock receptor.pdb "CCO" --pocket_residue A:145 --affinity
```

The module form
`python -m abcfold.boltz.dock_crystal` remains available for development
checkouts.

### Adding MMseqs2 MSAs and templates

To add MMseqs2 MSAs and templates to the AlphaFold3 input JSON, you can use the `mmseqs2msa`:

#### With Templates

To run the script with templates, use the following command:

```bash
uv run mmseqs2msa --input_json <input_json> --output_json <output_json> --templates --num_templates <num_templates>
```

- `<input_json>`: Path to the input AlphaFold3 JSON file.
- `<output_json>`: [optional] Path to the output JSON file (default: `<input_json_stem>`_mmseqs.json).
- `<num_templates>`: [optional] The number of templates to use (default: 20)
- `<mmseqs_database>`: [optional] The path to the database used by a local copy of MMSeqs2, provided mmseqs is installed, the inclusion of this flag allows MMseqs2 to be run locally.

> [!NOTE]
> If you need to install the mmseqs databases you can use setup_mmseqs_databases.sh
> This replicates the MMSeqs2 database setup from ColabFold

```
bash
MMSEQS_NO_INDEX=1 ./setup_mmseqs_databases.sh /path/to/db_folder
```


#### Without Templates

To run the script without templates, use the following command:

```bash
uv run mmseqs2msa --input_json <input_json> --output_json <output_json>
```

- `<input_json>`: Path to the input AlphaFold3 JSON file.
- `<output_json>`: [optional] Path to the output JSON file (default: `<input_json_stem>`_mmseqs.json).


### Adding custom templates

You may wish to add custom templates to your AlphaFold3 job, e.g. homologues which have yet to be deposited in the PDB. You can do so in two ways:

#### add_custom_template.py

If you just wish to add a custom template, you can use `custom_templates`:

```bash
uv run custom_templates --input_json <input_json> --output_json <output_json> --custom_template <custom_template> --custom_template_chain <custom_template_chain> --target_id <target_id>
```

- `<input_json>`: Path to the input AlphaFold3 JSON file.
- `<output_json>`: [optional] Path to the output JSON file (default: `<input_json_stem>`_custom_template.json).
- `<custom_template>`: [optional] Path to a custom template file in mmCIF format or a list of custom templates.
- `<custom_template_chain>`: [conditionally required] The chain ID of the chain to use in your custom template, only required if using a multi-chain template. If providing a list of custom templates, you will need to provide a list of custom template chains.
- `<target_id>`: [conditionally required] The ID of the sequence the custom template relates to, only required if modelling a complex. If providing a list of custom templates, you can provide a single target ID if they all relate to the same target. Otherwise, you should provide a list of target IDs corresponding to the list of custom templates.


#### add_mmseqs_msa.py

If you wish to add a custom template and generate an MMseqs2 MSA/templates, you can use `mmseqs2msa`:

```bash
uv run mmseqs2msa --input_json <input_json> --output_json <output_json> --templates --num_templates <num_templates> --custom_template <custom_template> --custom_template_chain <custom_template_chain> --target_id <target_id>
```

- `<input_json>`: Path to the input AlphaFold3 JSON file.
- `<output_json>`: [optional] Path to the output JSON file (default: `<input_json_stem>`_mmseqs.json).
- `<num_templates>`: [optional] The number of templates to use (default: 20)
- `<custom_template>`: [optional] Path to a custom template file in mmCIF format or a list of custom templates.
- `<custom_template_chain>`: [conditionally required] The chain ID of the chain to use in your custom template, only required if using a multi-chain template. If providing a list of custom templates, you will need to provide a list of custom template chains.
- `<target_id>`: [conditionally required] The ID of the sequence the custom template relates to, only required if modelling a complex. If providing a list of custom templates, you can provide a single target ID if they all relate to the same target. Otherwise, you should provide a list of target IDs corresponding to the list of custom templates.


### Possible Issues

#### Using `--target_id` with homo-oligomer

Below is an example of a hetero-3-mer. When modelling a homo-oligomer, id is given as a list, you should select 1 of the identifiers in the list.

```json
{
  "name": "7ZYH",
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "SNAESKIKDCPWYDRGFCKHGPLCRHRHTRRVICVNYLVGFCPEGPSCKFMHPRFELPMGTTEQ"
      }
    },
    {
      "protein": {
        "id": ["B", "C"],
        "sequence": "SNAGSINGVPLLEVDLDSFEDKPWRKPGADLSDYFNYGFNEDTWKAYCEKQKRIRMGLEVIPVTSTTNK"
      }
    }
  ],
  "modelSeeds": [1],
  "dialect": "alphafold3",
  "version": 1
}
```

If you want to add a custom template to the first sequence, you can use `--target_id A`. If you wish to add a custom template to the second sequence, use `--target_id B` or `--target_id C`.

#### Boltz limitations

If modelling multiple copies of the same sequence in Boltz, the input JSON must be set up as follows:

```json
{
  "name": "7ZYH",
  "sequences": [
    {
      "protein": {
        "id": ["A", "B"],
        "sequence": "SNAESKIKDCPWYDRGFCKHGPLCRHRHTRRVICVNYLVGFCPEGPSCKFMHPRFELPMGTTEQ"
      }
    },
  ],
  "modelSeeds": [1],
  "dialect": "alphafold3",
  "version": 1
}
```

If the identical sequences are given as seperate entities (as shown below) you will encounter an error.

```json
{
  "name": "7ZYH",
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "SNAESKIKDCPWYDRGFCKHGPLCRHRHTRRVICVNYLVGFCPEGPSCKFMHPRFELPMGTTEQ"
      }
    },
    {
      "protein": {
        "id": "B",
        "sequence": "SNAESKIKDCPWYDRGFCKHGPLCRHRHTRRVICVNYLVGFCPEGPSCKFMHPRFELPMGTTEQ"
      }
    }
  ],
  "modelSeeds": [1],
  "dialect": "alphafold3",
  "version": 1
}
```

Additionally, Boltz currently lack the ability to create linked-ligands and therefore covalent bonds between the chain/ligand will be missing.

#### OpenFold3 limitations

Currently OpenFold3 is unable to create bonded-pairs or alter the number of recycles. Therefore these options will be ignored if provided.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

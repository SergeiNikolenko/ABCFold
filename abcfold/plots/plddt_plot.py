# import numpy as np
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo

from abcfold.output.file_handlers import CifFile
from abcfold.output.utils import get_gap_indicies, insert_none_by_minus_one

logger = logging.getLogger("logger")


def plot_plddt(
    cif_models_dict: Dict[str, List[CifFile]],
    output_name: Union[str, Path],
    line_width: float = 1.6,
    dash: str = "dot",
    show: bool = False,
    chain_line_occupancy: float = 0.9,
    include_plotlyjs: bool = True,
) -> None:
    """
    Plots the pLDDT distribution of the models in the dictionary of cif models. Outputs
    and html file with the plot.

    Args:
        cif_models_dict: Dictionary of cif models to plot. The keys are the source of
            the models and the values are lists of CifFile objects.
            e.g. {"Alphafold3": [CifFile, CifFile, ...], "Boltz-1": [CifFile, ...],
            "Chai-1": [CifFile, ...]}

        output_name: Path to the output html file.
        line_width: Width of the lines in the plot.
        dash: Dash style of the lines in the plot.
        show: If True, the plot will be displayed in the browser.
        chain_line_occupancy: Opacity of the vertical lines that separate the chains.

    Returns:
        None

    Outputs:
        An html file with the plot.
    """

    fig = go.Figure()
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGrey")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGrey")

    colours = list(px.colors.qualitative.T10)
    method_colours = {
        "Alphafold3": px.colors.qualitative.Set1,
        "Boltz-1": px.colors.qualitative.Set2,
        "Chai-1": px.colors.qualitative.Prism,
    }

    line_ranges: dict = {}

    cif_models = [
        cif_file for cif_files in cif_models_dict.values() for cif_file in cif_files
    ]
    indicies = get_gap_indicies(*cif_models)
    indicies_index = 0

    for method, cif_models in cif_models_dict.items():

        for cif_model in cif_models:
            model_index = int(cif_model.name.split("_")[-1])
            color_list = method_colours.get(method, colours)
            color = color_list[model_index % len(color_list)]

            plddt = cif_model.residue_plddts

            if len(indicies) > 0:
                plddt = insert_none_by_minus_one(indicies[indicies_index], plddt)

            indicies_index += 1
            chain_ranges = {
                chain: len(plddt)
                for chain, plddt in cif_model.get_plddt_per_residue().items()
            }
            line_ranges = {
                chain: max(chain_ranges[chain], line_ranges.get(chain, 0))
                for chain in chain_ranges
            }

            trace = go.Scatter(
                x=list(range(len(plddt))),
                y=plddt,
                mode="lines",
                legendgroup=method,
                legendgrouptitle_text=Bold(method),
                name=f"Model {model_index + 1}",
                line=dict(dash=dash, width=line_width, color=color),
                visible=True,  # Ensure traces start as visible
                showlegend=True,
            )
            fig.add_trace(trace)

    counter = 0
    colour_index = 0
    for chain, chain_range in line_ranges.items():
        counter += chain_range
        chain_name = f"Chain {chain}"

        fig.add_vline(
            x=counter - 1,
            line=dict(color=colours[colour_index % len(colours)], dash="dash"),
            opacity=chain_line_occupancy,
            annotation_text=Bold(chain_name),
            annotation_font_size=15,
            annotation_position="top left",
            annotation_textangle=-90,
        )

        colour_index += 1
    # Create buttons for each model
    buttons = []
    num_models = len(cif_models_dict[next(iter(cif_models_dict))])

    # Add a button to show all traces
    buttons.append(
        dict(
            method="update",
            args=[{"visible": [True] * len(fig.data)}, {"showlegend": True}],
            label="All",
        )
    )

    # Add buttons for each individual model
    for model_index in range(num_models):
        button: Dict[str, Any] = dict(
            method="update",
            args=[
                {
                    "visible": [
                        i % num_models == model_index for i in range(len(fig.data))
                    ]
                },
                {"showlegend": True},
            ],
            label=f"Model {model_index + 1}",
        )
        for i in range(model_index, len(fig.data), num_models):
            button["args"][0]["visible"][i] = True
        buttons.append(button)

    # Add the updatemenu to the layout
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                buttons=buttons,
                direction="left",
                x=0.5,
                xanchor="center",
                y=-0.1,
                yanchor="top",
            )
        ],
        xaxis_title=Bold("Residue Number"),
        yaxis_title=Bold("pLDDT Score"),
        title=Bold("pLDDT Distribution"),
        plot_bgcolor="white",
    )

    if show:
        fig.show()

    output_name = Path(output_name)

    if output_name.suffix == "":
        output_name = output_name.with_suffix(".html")

    if include_plotlyjs:
        fig.write_html(str(output_name))
    else:
        div = pyo.plot(fig, include_plotlyjs=False, output_type="div")
        output_name = output_name.with_suffix(".div.html")
        with open(output_name, "w") as f:
            f.write(div)


def Bold(string):
    return f"<b>{string}</b>"

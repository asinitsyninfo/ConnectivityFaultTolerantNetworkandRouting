import io
import gradio as gr
import numpy as np
import pandas as pd
import tempfile
import plotly.graph_objects as go


from ui.components.twisters import get_twisters
from umst.algorithm import (
    get_probs,
    get_route_priority_list,
    get_umst_with_probs,
    umst,
)
from umst.visualization import (
    get_graph,
    get_graph_figure,
    get_graph_figure_abstract,
    get_umst_graph_abstract_figure,
    get_umst_graph_figure,
)


def plot_graph(
    adjanecy: np.ndarray,
    capitals: pd.DataFrame,
    plot_abstract_graph: bool,
) -> go.Figure:
    user_data_validation(adjanecy, capitals)
    graph = get_graph(adjanecy, capitals)
    if plot_abstract_graph:
        fig = get_graph_figure_abstract(graph)
    else:
        fig = get_graph_figure(graph)
    return fig


def plot_umst(
    adjanecy: np.ndarray,
    capitals: pd.DataFrame,
    k: int,
    show_initial_graph: bool,
    plot_abstract_graph: bool,
) -> tuple[go.Figure, str, gr.DownloadButton]:
    user_data_validation(adjanecy, capitals)
    if k > len(capitals.index) - 2:
        raise gr.Error(f"K should be less than {len(capitals.index) - 2}")

    init_graph = get_graph(adjanecy, capitals)
    umst_graph = umst(init_graph, k)
    if plot_abstract_graph:
        if show_initial_graph:
            fig = get_umst_graph_abstract_figure(init_graph, umst_graph)
        else:
            fig = get_graph_figure_abstract(umst_graph)
    else:
        if show_initial_graph:
            fig = get_umst_graph_figure(init_graph, umst_graph)
        else:
            fig = get_graph_figure(umst_graph)

    probs = get_probs(umst_graph, ones=True)
    umst_graph_with_probs = get_umst_with_probs(umst_graph, k, probs)
    route_prioirty_list = get_route_priority_list(init_graph, umst_graph_with_probs)
    download_button = get_download_file_button(route_prioirty_list.encode("utf-8"))
    return (fig, route_prioirty_list, download_button)


def user_data_validation(
    adjanecy: np.ndarray,
    capitals: pd.DataFrame,
):
    if adjanecy.size < 4:
        raise gr.Error("matrix is empty")
    if len(capitals.index) < 2:
        raise gr.Error("capitals is empty")
    if adjanecy.shape[0] != adjanecy.shape[1]:
        raise gr.Error("matrix is not square")
    if adjanecy.shape[0] != len(capitals.index):
        raise gr.Error(f"different sizes: {adjanecy.shape[0]} / {len(capitals.index)}")


def get_download_file_button(text: bytes) -> gr.DownloadButton:
    with tempfile.NamedTemporaryFile(
        prefix="routes", suffix=".txt", delete=False
    ) as tmp:
        tmp.write(text)
        return gr.DownloadButton(value=tmp.name, interactive=True)


def get_gradio_app() -> gr.Blocks:
    with gr.Blocks(css=""".gradio-container {margin: 0 !important};""") as app:
        total_nodes_amount = gr.State(None)

        with gr.Row():
            with gr.Column(scale=4):
                capitals_df, matrix_df, selected_k, plot_abstract_graph = get_twisters(
                    total_nodes_amount
                )
            with gr.Column(scale=10):
                with gr.Tabs():
                    with gr.Tab("Initial Graph"):
                        initial_graph_plot = gr.Plot()
                        plot_initial_btn = gr.Button("Plot graph")
                    with gr.Tab("UMST Graph"):
                        umst_graph_plot = gr.Plot()
                        show_initial_graph = gr.Checkbox(
                            value=False, label="Show initial graph"
                        )
                        calc_umst_btn = gr.Button("Calculate UMST")
                        routing_priority_table = gr.Textbox(
                            label="Routing priority table",
                            info="Calculated based on UMST graph",
                            max_lines=20,
                            show_copy_button=True,
                        )
                        download_routing_priority_table = gr.DownloadButton(
                            "Download routing priority table", interactive=False
                        )

        plot_initial_btn.click(
            plot_graph,
            inputs=[
                matrix_df,
                capitals_df,
                plot_abstract_graph,
            ],
            outputs=initial_graph_plot,
        )
        calc_umst_btn.click(
            plot_umst,
            inputs=[
                matrix_df,
                capitals_df,
                selected_k,
                show_initial_graph,
                plot_abstract_graph,
            ],
            outputs=[
                umst_graph_plot,
                routing_priority_table,
                download_routing_priority_table,
            ],
        )

    return app

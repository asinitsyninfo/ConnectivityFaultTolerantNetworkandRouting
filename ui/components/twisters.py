import gradio as gr
import pandas as pd
import numpy as np

from ui.constants import CAPITALS_HEADERS

EXAMPLE_NODES = [
    ["London", 51.5074, -0.1278, 8],
    ["Paris", 48.8566, 2.3522, 6],
    ["Berlin", 52.5200, 13.4050, 5],
    ["Rome", 41.9028, 12.4964, 7],
    ["Amsterdam", 52.3676, 4.9041, 0.87],
    ["Brussels", 50.8503, 4.3517, 1.2],
    ["Copenhagen", 55.6761, 12.5683, 0.63],
    ["Stockholm", 59.3293, 18.0686, 0.97],
    ["Oslo", 59.9139, 10.7522, 0.68],
    ["Helsinki", 60.1695, 24.9354, 0.65],
]
EXAMPLE_NODES_LEN = len(EXAMPLE_NODES)


def set_example_capitals() -> (
    tuple[
        pd.DataFrame,
        gr.Slider,
        int,
    ]
):
    N = EXAMPLE_NODES_LEN
    return (
        pd.DataFrame(
            EXAMPLE_NODES,
            columns=CAPITALS_HEADERS,
        ),
        gr.Slider(
            maximum=N,
            interactive=True,
        ),
        N,
    )


def set_example_matrix() -> np.ndarray:
    N = EXAMPLE_NODES_LEN
    arr = np.ones((N, N))
    arr = np.triu(arr, 0)
    return arr


def load_capitals(
    filepath: str, total_nodes_amount: int
) -> tuple[
    pd.DataFrame,
    gr.Slider,
]:
    df = pd.read_csv(filepath)
    total_nodes_amount = len(df)

    return (
        df,
        gr.Slider(
            maximum=total_nodes_amount,
            interactive=True,
        ),
        total_nodes_amount,
    )


def load_matrix(filepath: str) -> np.ndarray:
    arr = np.loadtxt(filepath, delimiter=",")
    return arr


def get_twisters(
    total_nodes_amount: gr.State,
    capital_headers: tuple[str] = CAPITALS_HEADERS,
) -> tuple[pd.DataFrame, np.ndarray]:
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                selected_k = gr.Slider(
                    minimum=0,
                    maximum=0,
                    value=0,
                    step=1,
                    label="Number of broken nodes (K)",
                    interactive=False,
                )
            with gr.Column():
                plot_abstract_graph = gr.Checkbox(label="Plot abstract graphs")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    upload_capitals_button = gr.UploadButton(
                        "🏙️ Upload your capitals file",
                        file_count="single",
                        file_types=[".csv"],
                    )
                    example_capitals_button = gr.Button("Use default capitals")
            with gr.Column(scale=4):
                capitals_df = gr.DataFrame(
                    label="Capitals",
                    interactive=False,
                    headers=capital_headers,
                    col_count=(len(capital_headers), "fixed"),
                )
            example_capitals_button.click(
                set_example_capitals,
                outputs=[
                    capitals_df,
                    selected_k,
                    total_nodes_amount,
                ],
            )
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    upload_matrix_button = gr.UploadButton(
                        "📊 Upload your matrix file",
                        file_count="single",
                        file_types=[".csv"],
                    )
                    example_matrix_button = gr.DownloadButton("Use default matrix")
            with gr.Column(scale=4):
                matrix_df = gr.DataFrame(
                    label="Adjacency matrix",
                    interactive=False,
                    headers=None,
                    type="numpy",
                )
            example_matrix_button.click(set_example_matrix, None, matrix_df)

        upload_capitals_button.upload(
            load_capitals,
            [upload_capitals_button, total_nodes_amount],
            [capitals_df, selected_k, total_nodes_amount],
        )
        upload_matrix_button.upload(
            load_matrix,
            upload_matrix_button,
            matrix_df,
        )
    return capitals_df, matrix_df, selected_k, plot_abstract_graph

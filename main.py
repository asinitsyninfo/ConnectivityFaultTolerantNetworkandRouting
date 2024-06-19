
import gradio as gr

from ui.root import get_gradio_app


if __name__ == "__main__":
    app = get_gradio_app()
    app.launch()

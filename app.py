import gradio as gr
import torch
from api import FlexSED
import tempfile
import os

# Load model once on startup
flexsed = FlexSED(device="cuda" if torch.cuda.is_available() else "cpu")

def run_flexsed(audio_file, event_list):
    """
    Run inference using FlexSED and return prediction plot.
    """
    if not audio_file:
        return None

    # Split events by semicolon or comma
    events = [e.strip() for e in event_list.split(";") if e.strip()]
    if not events:
        return None

    # Run inference
    preds = flexsed.run_inference(audio_file, events)

    # Generate visualization
    output_fname = os.path.join(tempfile.gettempdir(), "flexsed_output")
    flexsed.to_multi_plot(preds, events, fname=output_fname)
    plot_path = f"{output_fname}.png"

    return plot_path


# App layout
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as app:
    # Header
    gr.Markdown("""
    ## 🎧 FlexSED: A Flexible Open-Vocabulary Sound Event Detection System

    👋 Welcome to the **FlexSED live demo** — explore **prompt-guided sound event detection** in real audio clips.

    🔗 Learn more on the [FlexSED GitHub Repository](https://github.com/JHU-LCAP/FlexSED)
    """)

    gr.Markdown("### 🔍 Upload or choose an example below to detect sound events:")

    with gr.Row():
        # Left column: Inputs
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="🎵 Upload Audio (.wav)")
            text_input = gr.Textbox(label="Event list (semicolon-separated)", value="Male speech; Door; Dog; Laughter")

            with gr.Row():
                detect_btn = gr.Button("🎯 Detect", variant="primary")
                clear_btn = gr.Button("🧹 Clear")

        # Right column: Output
        with gr.Column(scale=1):
            image_output = gr.Image(label="Prediction Plot", show_label=True, elem_id="output-image")
            gr.Examples(
                examples=[
                    ["example.wav", "Male speech; Door; Dog; Laughter"],
                    ["example2.wav", "Male speech; Bee; Gunshot, gunfire"],
                ],
                inputs=[audio_input, text_input],
                label="Example Audios"
            )

    # Function bindings
    detect_btn.click(run_flexsed, inputs=[audio_input, text_input], outputs=image_output)
    clear_btn.click(lambda: (None, "Male speech; Door; Dog; Laughter"), outputs=[audio_input, text_input])


if __name__ == "__main__":
    app.launch(share=True)

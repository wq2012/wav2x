import gradio as gr
import os
from sidlingvo import wav_to_dvector
from huggingface_hub import hf_hub_download

title = "Speaker Recognition Demo"

description = """
A demo of conformer-based speaker recognition.
Paper: https://arxiv.org/abs/2104.02125
Model: https://huggingface.co/tflite-hub/conformer-speaker-encoder
"""

repo_id = "tflite-hub/conformer-speaker-encoder"
model_path = "models"
hf_hub_download(repo_id=repo_id, filename="vad_long_model.tflite", local_dir=model_path)
hf_hub_download(repo_id=repo_id, filename="vad_long_mean_stddev.csv", local_dir=model_path)
hf_hub_download(repo_id=repo_id, filename="conformer_tisid_medium.tflite", local_dir=model_path)

runner = wav_to_dvector.WavToDvectorRunner(
    vad_model_file=os.path.join(model_path, "vad_long_model.tflite"),
    vad_mean_stddev_file=os.path.join(model_path, "vad_long_mean_stddev.csv"),
    tisid_model_file=os.path.join(model_path, "conformer_tisid_medium.tflite"))

def predict(enroll_audio, test_audio):
    score = runner.compute_score([enroll_audio], test_audio)
    return "Speaker similarity score: " + str(score)

if __name__ == "__main__":
    demo = gr.Interface(
        fn=predict,
        inputs=[gr.Audio(type="filepath"), gr.Audio(type="filepath")],
        outputs="text",
        title=title,
        description=description,)
    demo.launch()
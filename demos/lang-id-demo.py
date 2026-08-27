"""A demo of conformer-based spoken language identification."""

import os
from huggingface_hub import hf_hub_download
import numpy as np
from wav2x import wav_to_lang

title = "Spoken Language Identification"

description = """
A demo of conformer-based spoken language identification.
Paper: https://arxiv.org/abs/2202.12163
Model: https://huggingface.co/tflite-hub/conformer-lang-id
"""

repo_id = "tflite-hub/conformer-lang-id"
model_path = "models"
hf_hub_download(
    repo_id=repo_id, filename="vad_short_model.tflite", local_dir=model_path
)
hf_hub_download(
    repo_id=repo_id,
    filename="vad_short_mean_stddev.csv",
    local_dir=model_path,
)
hf_hub_download(
    repo_id=repo_id,
    filename="conformer_langid_medium.tflite",
    local_dir=model_path,
)

runner = wav_to_lang.WavToLangRunner(
    vad_model_file=os.path.join(model_path, "vad_short_model.tflite"),
    vad_mean_stddev_file=os.path.join(model_path, "vad_short_mean_stddev.csv"),
    langid_model_file=os.path.join(
        model_path, "conformer_langid_medium.tflite"
    ),
)


def predict(wav_file):
  top_lang, probs = runner.wav_to_lang(wav_file)
  top_lang_prob = np.max(probs)
  return (
      "Predicted language: "
      + top_lang
      + "\nProbability: "
      + str(top_lang_prob)
  )


if __name__ == "__main__":
  import gradio as gr
  demo = gr.Interface(
      fn=predict,
      inputs=gr.Audio(type="filepath"),
      outputs="text",
      title=title,
      description=description,
  )
  demo.launch()

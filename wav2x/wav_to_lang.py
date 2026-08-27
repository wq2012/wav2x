"""A library to run lang-id TFLite model inference."""

import dataclasses
import os
from typing import Tuple
import colortimelog
import numpy as np
from wav2x import fe_utils
from wav2x import language_map


@dataclasses.dataclass
class WavToLangRunner:
  """Runner for spoken language identification."""

  # Path to the VAD TFLite model.
  vad_model_file: str = ""
  # Path to the VAD mean and stddev CSV file.
  vad_mean_stddev_file: str = ""
  # Path to the lang-id TFLite model.
  langid_model_file: str = ""

  vad_threshold: float = 0.1
  vad_cluster_id: int = 2
  vad_num_clusters: int = 16

  def __post_init__(self):
    self.vad_model = (
        fe_utils.load_tflite_model(self.vad_model_file)
        if self.vad_model_file
        else None
    )
    self.langid_model = (
        fe_utils.load_tflite_model(self.langid_model_file)
        if self.langid_model_file
        else None
    )
    self.vad_mean_stddev = (
        fe_utils.read_mean_stddev_csv(self.vad_mean_stddev_file)
        if self.vad_mean_stddev_file
        else None
    )
    self.feature_extractor = fe_utils.LogMelFeatureExtractor()

  @classmethod
  def from_pretrained(
      cls,
      model_dir: str = "models",
      download_if_missing: bool = True,
      vad_threshold: float = 0.1,
  ) -> "WavToLangRunner":
    """Instantiates a runner, downloading models if needed."""
    vad_model = os.path.join(model_dir, "vad_short_model.tflite")
    vad_csv = os.path.join(model_dir, "vad_short_mean_stddev.csv")
    langid_model = os.path.join(model_dir, "conformer_langid_medium.tflite")

    needed_files = [vad_model, vad_csv, langid_model]
    if download_if_missing and not all(os.path.exists(f) for f in needed_files):
      from huggingface_hub import hf_hub_download
      os.makedirs(model_dir, exist_ok=True)
      repo_id = "tflite-hub/conformer-lang-id"
      hf_hub_download(
          repo_id=repo_id,
          filename="vad_short_model.tflite",
          local_dir=model_dir,
      )
      hf_hub_download(
          repo_id=repo_id,
          filename="vad_short_mean_stddev.csv",
          local_dir=model_dir,
      )
      hf_hub_download(
          repo_id=repo_id,
          filename="conformer_langid_medium.tflite",
          local_dir=model_dir,
      )

    return cls(
        vad_model_file=vad_model,
        vad_mean_stddev_file=vad_csv,
        langid_model_file=langid_model,
        vad_threshold=vad_threshold,
    )

  def wav_to_lang(self, audio_file: str) -> Tuple[str, np.ndarray]:
    """Run lang-id model on audio file.

    Args:
      audio_file: Path to audio file.

    Returns:
      Tuple of (predicted language code, output probability array for last
      frame).
    """
    input_signal = fe_utils.get_int_samples(audio_file)
    return self.samples_to_lang(input_signal)

  @colortimelog.timefunc
  def samples_to_lang(
      self, input_signal: np.ndarray
  ) -> Tuple[str, np.ndarray]:
    """Run lang-id model on int16 samples."""
    if self.vad_model is None:
      raise ValueError("VAD model is not loaded.")
    if self.langid_model is None:
      raise ValueError("Lang-id model is not loaded.")
    if self.vad_mean_stddev is None:
      raise ValueError("VAD mean and stddev file is not loaded.")

    features_batch = self.feature_extractor.extract(input_signal)
    features = features_batch[0]

    self.vad_model.reset_all_variables()
    _, features_after_vad = fe_utils.apply_vad(
        features,
        self.vad_model,
        self.vad_mean_stddev,
        self.vad_threshold,
        self.vad_cluster_id,
        self.vad_num_clusters,
    )

    self.langid_model.reset_all_variables()
    langid_outputs = fe_utils.run_multi_input_model(
        self.langid_model, features_after_vad
    )

    last_langid_frame = langid_outputs[-1, :]
    top_class = int(np.argmax(last_langid_frame))
    language_code = language_map.ID_TO_LANG.get(
        top_class, f"unknown_{top_class}"
    )

    return language_code, last_langid_frame

"""A library to run speaker-id TFLite model inference."""

import dataclasses
import os
from typing import Dict, List, Tuple
import colortimelog
import numpy as np
from wav2x import fe_utils


def aggregate_dvectors(dvectors: List[np.ndarray]) -> np.ndarray:
  """Aggregate dvectors from multiple utterances."""
  normalized_dvectors = [
      dvector / np.linalg.norm(dvector) for dvector in dvectors
  ]
  stacked_dvectors = np.stack(normalized_dvectors, axis=0)
  return np.mean(stacked_dvectors, axis=0, keepdims=False)


def compute_cosine(vec1: np.ndarray, vec2: np.ndarray) -> float:
  """Compute cosine similarity between two vectors."""
  norm1 = np.linalg.norm(vec1)
  norm2 = np.linalg.norm(vec2)
  if norm1 == 0.0 or norm2 == 0.0:
    return 0.0
  return float(np.inner(vec1, vec2) / (norm1 * norm2))


@dataclasses.dataclass
class WavToDvectorRunner:
  """Runner for speaker-id d-vector extraction and verification."""

  # Path to the VAD TFLite model.
  vad_model_file: str = ""
  # Path to the VAD mean and stddev CSV file.
  vad_mean_stddev_file: str = ""
  # Path to the speaker-id TFLite model.
  tisid_model_file: str = ""

  vad_threshold: float = 0.1
  vad_cluster_id: int = 2
  vad_num_clusters: int = 16

  enrolled_speakers: Dict[str, np.ndarray] = dataclasses.field(
      default_factory=dict,
  )

  def __post_init__(self):
    self.vad_model = (
        fe_utils.load_tflite_model(self.vad_model_file)
        if self.vad_model_file
        else None
    )
    self.tisid_model = (
        fe_utils.load_tflite_model(self.tisid_model_file)
        if self.tisid_model_file
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
  ) -> "WavToDvectorRunner":
    """Instantiates a runner, downloading models if needed."""
    vad_model = os.path.join(model_dir, "vad_long_model.tflite")
    vad_csv = os.path.join(model_dir, "vad_long_mean_stddev.csv")
    tisid_model = os.path.join(model_dir, "conformer_tisid_medium.tflite")

    needed_files = [vad_model, vad_csv, tisid_model]
    if download_if_missing and not all(os.path.exists(f) for f in needed_files):
      from huggingface_hub import hf_hub_download
      os.makedirs(model_dir, exist_ok=True)
      repo_id = "tflite-hub/conformer-speaker-encoder"
      hf_hub_download(
          repo_id=repo_id, filename="vad_long_model.tflite", local_dir=model_dir
      )
      hf_hub_download(
          repo_id=repo_id,
          filename="vad_long_mean_stddev.csv",
          local_dir=model_dir,
      )
      hf_hub_download(
          repo_id=repo_id,
          filename="conformer_tisid_medium.tflite",
          local_dir=model_dir,
      )

    return cls(
        vad_model_file=vad_model,
        vad_mean_stddev_file=vad_csv,
        tisid_model_file=tisid_model,
        vad_threshold=vad_threshold,
    )

  def wav_to_dvector(self, audio_file: str) -> np.ndarray:
    """Run speaker-id model on audio file."""
    input_signal = fe_utils.get_int_samples(audio_file)
    return self.samples_to_dvector(input_signal)

  @colortimelog.timefunc
  def samples_to_dvector(self, input_signal: np.ndarray) -> np.ndarray:
    """Run speaker-id model on int16 samples."""
    if self.vad_model is None:
      raise ValueError("VAD model is not loaded.")
    if self.tisid_model is None:
      raise ValueError("Speaker-id model is not loaded.")
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

    self.tisid_model.reset_all_variables()
    dvectors = fe_utils.run_multi_input_model(
        self.tisid_model, features_after_vad
    )
    return dvectors

  def compute_score(
      self, enroll_audio_list: List[str], test_audio: str
  ) -> float:
    """Compute cosine similarity score between enrolled audio and test audio."""
    enroll_dvectors = [
        self.wav_to_dvector(enroll_audio)[-1, :]
        for enroll_audio in enroll_audio_list
    ]
    aggregate_enroll_dvector = aggregate_dvectors(enroll_dvectors)
    test_dvector = self.wav_to_dvector(test_audio)[-1, :]
    return compute_cosine(aggregate_enroll_dvector, test_dvector)

  def enroll_speaker(self, name: str, enroll_audio_list: List[str]) -> None:
    """Enroll a speaker with a list of audio recordings."""
    if not name:
      raise ValueError("Name cannot be empty.")
    enroll_dvectors = [
        self.wav_to_dvector(enroll_audio)[-1, :]
        for enroll_audio in enroll_audio_list
    ]
    self.enrolled_speakers[name] = aggregate_dvectors(enroll_dvectors)

  def clear_enrollment(self) -> None:
    """Clear all enrolled speakers."""
    self.enrolled_speakers = {}

  def identify_speaker(
      self, test_audio: str, threshold: float
  ) -> Tuple[str, float]:
    """Identify the speaker of the test audio from all enrolled speakers."""
    test_dvector = self.wav_to_dvector(test_audio)[-1, :]
    max_score = -1.0
    max_name = ""
    for name, enroll_dvector in self.enrolled_speakers.items():
      score = compute_cosine(enroll_dvector, test_dvector)
      if score > max_score:
        max_score = score
        max_name = name
    if max_score < threshold:
      return "", max_score
    return max_name, max_score

"""A library to run lang-id TFLite model inference."""

import dataclasses
import numpy as np
import colortimelog
import fe_utils
import language_map


@dataclasses.dataclass
class WavToLangRunner:
  """WavToLangRunner."""

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
    self.vad_model = fe_utils.load_tflite_model(self.vad_model_file)
    self.langid_model = fe_utils.load_tflite_model(self.langid_model_file)
    self.vad_mean_stddev = fe_utils.read_mean_stddev_csv(
        self.vad_mean_stddev_file
    )
    self.feature_extractor = fe_utils.LogMelFeatureExtractor()

  def wav_to_lang(self, audio_file: str) -> tuple[str, np.ndarray]:
    """Run lang-id model on wav file."""
    input_signal = fe_utils.get_int_samples(audio_file)
    probs = self.samples_to_lang(input_signal)
    # Take the last frame's prediction
    last_frame = probs[-1]
    # last_frame shape is (batch_size, num_classes). 
    # We want class with highest score in the last batch (or average).
    # Assuming batch_size=1 for simplicity in logic matching original.
    top_class = int(np.argmax(last_frame)) 
    # If flattened argmax is > num_classes, it means we have batch>1.
    # We should average or take first? 
    # Let's take the first element of the batch if batch>1.
    if len(last_frame.shape) > 1 and last_frame.shape[0] > 1:
        top_class = int(np.argmax(last_frame[0]))
    elif len(last_frame.shape) > 1:
        top_class = int(np.argmax(last_frame))
        
    try:
      language_code = language_map.ID_TO_LANG[top_class]
    except KeyError:
      print(f"ERROR: top_class {top_class} not in language_map! raw_output_last={last_frame}")
      raise
    
    return language_code, probs

  @colortimelog.timefunc
  def samples_to_lang(self, input_signal: np.ndarray) -> np.ndarray:
    """Computes probabilities (or logits) for languages."""
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
    language_code = language_map.ID_TO_LANG[int(np.argmax(last_langid_frame))]

    return language_code, last_langid_frame

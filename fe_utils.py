"""Utilities for the audio feature frontend."""

import math
import dataclasses
import librosa
from scipy.io import wavfile
import tensorflow.compat.v2 as tf
import numpy as np
import colortimelog


@dataclasses.dataclass
class LogMelFeatureExtractor:
    """Log-Mel Feature Extractor replacing Lingvo's MelAsrFrontend."""
    frame_size_ms: float = 32.0
    frame_step_ms: float = 10.0
    num_bins: int = 128
    sample_rate: float = 16000.0
    lower_edge_hertz: float = 125.0
    upper_edge_hertz: float = 7500.0
    preemph: float = 0.97
    noise_scale: float = 8.0
    stack_left_context: int = 3
    stack_right_context: int = 0
    frame_stride: int = 3

    def __post_init__(self):
        self.frame_length = int(self.frame_size_ms * self.sample_rate / 1000)
        self.frame_step = int(self.frame_step_ms * self.sample_rate / 1000)
        self.fft_length = 512  # 2 ** ceil(log2(frame_length))
        
        # Ensure fft_length is sufficient
        if self.fft_length < self.frame_length:
            self.fft_length = 2 ** int(np.ceil(np.log2(self.frame_length)))

    def extract(self, input_signal: np.ndarray) -> np.ndarray:
        """Extracts features from input signal.
        
        Args:
            input_signal: (batch, time) int16 or float array.
            
        Returns:
            (batch, time, feature_dim) float32 array.
        """
        # Convert to tensor if not already
        signal = tf.convert_to_tensor(input_signal, dtype=tf.float32)
        
        # Add noise (dithering)
        if self.noise_scale > 0:
            noise = tf.random.uniform(tf.shape(signal), minval=-self.noise_scale, maxval=self.noise_scale)
            signal = signal + noise
            
        # Pre-emphasis
        if self.preemph > 0:
            # Shift signal by 1 to the right to get s[t-1]
            # Pad the left with 0
            padded_signal = tf.pad(signal, [[0, 0], [1, 0]])
            signal = signal - self.preemph * padded_signal[:, :-1]
            
        # STFT
        stfts = tf.signal.stft(
            signal,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.fft_length,
            window_fn=tf.signal.hann_window,
            pad_end=False
        )
        spectrograms = tf.abs(stfts)
        
        # Mel Filterbank
        num_spectrogram_bins = self.fft_length // 2 + 1
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            self.num_bins,
            num_spectrogram_bins,
            self.sample_rate,
            self.lower_edge_hertz,
            self.upper_edge_hertz
        )
        
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        
        # Log (stabilized)
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        
        # Stacking and Subsampling
        return self._stack_and_subsample(log_mel_spectrograms)
        
    def _stack_and_subsample(self, features: tf.Tensor) -> np.ndarray:
        """Stacks frames and subsamples."""
        # features: [batch, time, bins]
        
        stack_width = self.stack_left_context + 1 + self.stack_right_context
        
        # Pad features for left context
        padded_features = tf.pad(features, [[0, 0], [self.stack_left_context, self.stack_right_context], [0, 0]])
        
        # We construct the stacked frames
        time_steps = tf.shape(features)[1]
        
        stacked_cols = []
        for i in range(stack_width):
            # We slice from i to i + time_steps
            stacked_cols.append(padded_features[:, i : i + time_steps, :])
            
        stacked = tf.concat(stacked_cols, axis=2)
        
        # Subsample
        subsampled = stacked[:, ::self.frame_stride, :]
        
        return subsampled.numpy()


def read_mean_stddev_csv(csv_file: str) -> tuple[np.ndarray, np.ndarray]:
  """Reads mean and stddev from a CSV file."""
  with open(csv_file) as f:
    csv_data = np.genfromtxt(f, delimiter=",")
  return csv_data[0, :], csv_data[1, :]


@colortimelog.timefunc
def get_int_samples(file_name: str) -> np.ndarray:
  with open(file_name, "rb") as f:
    sample_rate, int_samples = wavfile.read(f)

  if sample_rate != 16000:
    samples = int_samples / 32768.0
    samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
    int_samples = (samples * 32768.0).astype(np.int16)

  return np.expand_dims(int_samples, axis=0)


def add_cluster_id(
    features: np.ndarray, cluster_id: int, num_clusters: int
) -> np.ndarray:
  if not num_clusters:
    return features
  cluster_ids = np.zeros((features.shape[0], num_clusters))
  cluster_ids[:, cluster_id] = 1
  return np.concatenate([features, cluster_ids], axis=1).astype(np.float32)


def normalize_features(
    features: np.ndarray, mean_stddev: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
  vad_mean, vad_std_dev = mean_stddev
  mean_vec = np.expand_dims(np.repeat(vad_mean, 4), axis=0)
  std_dev_vec = np.expand_dims(np.repeat(vad_std_dev, 4), axis=0)
  features = (features - mean_vec) / std_dev_vec
  return features.astype(np.float32)


@colortimelog.timefunc
def load_tflite_model(model_path: str) -> tf.lite.Interpreter:
  """Reads a serialized TFLite model and returns the Interpreter object."""
  # Read model.
  with open(model_path, "rb") as file_object:
    model_content = file_object.read()
  tflite_model = tf.lite.Interpreter(model_content=model_content)
  tflite_model.allocate_tensors()
  tflite_model.reset_all_variables()
  return tflite_model


def run_single_input_model_one_step(
    model: tf.lite.Interpreter, input_data: np.ndarray
) -> np.ndarray:
  """Runs given TFLite model (as interpreter) on data for a single step."""
  model.set_tensor(
      model.get_input_details()[0]["index"],
      input_data,
  )
  model.invoke()
  return model.get_tensor(model.get_output_details()[0]["index"])


def print_model_info(model: tf.lite.Interpreter) -> None:
  """Returns input and output tensor details."""
  model.reset_all_variables()
  input_details = model.get_input_details()
  output_details = model.get_output_details()
  print("input_details:", input_details)
  print("output_details:", output_details)


def apply_vad(
    features: np.ndarray,
    vad_model: tf.lite.Interpreter,
    mean_stddev: tuple[np.ndarray, np.ndarray],
    threshold: float,
    cluster_id: int = 2,
    num_clusters: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
  """Applies VAD to the features."""
  normalized_features = normalize_features(features, mean_stddev)
  normalized_features = add_cluster_id(
      normalized_features, cluster_id, num_clusters
  )

  vad_decisions = []
  for t in range(normalized_features.shape[0]):
    features_t = normalized_features[t, :]
    features_t = np.expand_dims(features_t, axis=0)
    output = run_single_input_model_one_step(vad_model, features_t)
    vad_score = math.exp(output[0, 0])
    vad_decisions.append(vad_score > threshold)
  vad_decisions = np.array(vad_decisions)

  features_after_vad = features[vad_decisions, :]
  return vad_decisions, features_after_vad


def run_multi_input_model_one_step(
    model: tf.lite.Interpreter, input_data: np.ndarray, states: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray]]:
  """Runs given TFLite model (as interpreter) on data for a single step."""
  model.set_tensor(
      model.get_input_details()[0]["index"],
      input_data,
  )
  for i, state in enumerate(states):
    model.set_tensor(
        model.get_input_details()[i + 1]["index"],
        state,
    )
  model.invoke()
  output_data = model.get_tensor(model.get_output_details()[0]["index"])
  output_states = []
  for i in range(len(states)):
    output_states.append(
        model.get_tensor(model.get_output_details()[i + 1]["index"])
    )
  return output_data, output_states


def run_multi_input_model(
    model: tf.lite.Interpreter, features: np.ndarray, batch_size: int = 2
) -> np.ndarray:
  """Runs given TFLite model (as interpreter) on data for a sequence."""
  states = []
  outputs = []
  for i in range(len(model.get_input_details()) - 1):
    states.append(
        np.zeros(
            shape=model.get_input_details()[i + 1]["shape"],
            dtype=model.get_input_details()[i + 1]["dtype"],
        )
    )
  for t in range(int(features.shape[0] / batch_size)):
    start_index = t * batch_size
    features_t = features[start_index: start_index + batch_size, :]
    output, states = run_multi_input_model_one_step(model, features_t, states)
    outputs.append(output)
  return np.concatenate(outputs, axis=0)

"""Utilities for the audio feature frontend."""

import dataclasses
import math
from typing import Tuple
import colortimelog
import librosa
import numpy as np
from scipy.io import wavfile
import soundfile as sf
import tensorflow.compat.v2 as tf


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
  noise_scale: float = 0.0
  pad_end: bool = False
  fft_overdrive: bool = True
  output_floor: float = 1.0
  stack_left_context: int = 3
  stack_right_context: int = 0
  frame_stride: int = 3

  def __post_init__(self):
    self.frame_length = int(
        round(self.sample_rate * self.frame_size_ms / 1000.0)
    )
    self.frame_step = int(
        round(self.sample_rate * self.frame_step_ms / 1000.0)
    )
    self._window_size = self.frame_length
    self._framed_size = (
        self.frame_length + 1 if self.preemph > 0.0 else self.frame_length
    )

    min_fft = 512
    p2 = 1 << (self.frame_length - 1).bit_length()
    self.fft_length = max(min_fft, p2)
    if self.fft_overdrive:
      self.fft_length *= 2

    self._mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=self.num_bins,
        num_spectrogram_bins=self.fft_length // 2 + 1,
        sample_rate=self.sample_rate,
        lower_edge_hertz=self.lower_edge_hertz,
        upper_edge_hertz=self.upper_edge_hertz,
        dtype=tf.float32,
    )

  def extract(self, input_signal: np.ndarray) -> np.ndarray:
    """Extracts log-mel filterbank features from an input audio signal.

    Args:
      input_signal: (batch, time) array of int16 or float32 PCM samples
        scaled to [-32768, 32767].

    Returns:
      (batch, time, feature_dim) float32 numpy array.
    """
    signal = tf.cast(input_signal, tf.float32)

    # Frame the signal. Shape: [batch, time, _framed_size]
    framed_signal = tf.signal.frame(
        signal,
        frame_length=self._framed_size,
        frame_step=self.frame_step,
        pad_end=self.pad_end,
    )

    # Pre-emphasis across consecutive samples in each frame.
    if self.preemph > 0.0:
      preemphasized = (
          framed_signal[:, :, 1:] - self.preemph * framed_signal[:, :, :-1]
      )
    else:
      preemphasized = framed_signal

    # Optional noise dithering.
    if self.noise_scale > 0.0:
      noise = tf.random.normal(
          tf.shape(preemphasized),
          stddev=self.noise_scale,
          mean=0.0,
      )
      windowed_input = preemphasized + noise
    else:
      windowed_input = preemphasized

    # Apply Hann window.
    window = tf.signal.hann_window(self._window_size, dtype=signal.dtype)
    windowed = windowed_input * window

    # RFFT and magnitude spectrogram.
    rfft = tf.signal.rfft(windowed, [self.fft_length])
    magnitude = tf.abs(rfft)

    # Mel filterbank.
    mel = tf.matmul(magnitude, self._mel_weight_matrix)

    # Log mel with flooring.
    log_mel = tf.math.log(tf.maximum(self.output_floor, mel))

    # Pad context by replicating edge frames (matching Lingvo frontend).
    if self.stack_left_context > 0:
      log_mel = tf.concat(
          [log_mel[:, 0:1, :]] * self.stack_left_context + [log_mel],
          axis=1,
      )
    if self.stack_right_context > 0:
      log_mel = tf.concat(
          [log_mel] + [log_mel[:, -1:, :]] * self.stack_right_context,
          axis=1,
      )

    # Stacking and sub-sampling.
    stack_size = 1 + self.stack_left_context + self.stack_right_context
    if stack_size > 1 or self.frame_stride > 1:
      stacked = tf.signal.frame(
          signal=log_mel,
          frame_length=stack_size,
          frame_step=self.frame_stride,
          pad_end=False,
          axis=1,
      )
      batch_size = tf.shape(stacked)[0]
      num_frames = tf.shape(stacked)[1]
      features = tf.reshape(
          stacked, [batch_size, num_frames, stack_size * self.num_bins]
      )
    else:
      features = log_mel

    return features.numpy()


def read_mean_stddev_csv(csv_file: str) -> Tuple[np.ndarray, np.ndarray]:
  """Reads mean and stddev from a CSV file."""
  with open(csv_file) as f:
    csv_data = np.genfromtxt(f, delimiter=",")
  return csv_data[0, :], csv_data[1, :]


@colortimelog.timefunc
def get_int_samples(file_name: str) -> np.ndarray:
  """Reads an audio file and returns [1, num_samples] int16 array at 16kHz."""
  try:
    data, sample_rate = sf.read(file_name)
    if data.ndim > 1:
      data = np.mean(data, axis=1)
  except Exception:
    with open(file_name, "rb") as f:
      sample_rate, data = wavfile.read(f)
    if data.ndim > 1:
      data = np.mean(data, axis=1)
    if data.dtype == np.int16:
      data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
      data = data.astype(np.float32) / 2147483648.0

  if sample_rate != 16000:
    data = librosa.resample(data, orig_sr=sample_rate, target_sr=16000)

  int_samples = np.clip(data * 32768.0, -32768, 32767).astype(np.int16)
  return np.expand_dims(int_samples, axis=0)


def add_cluster_id(
    features: np.ndarray, cluster_id: int, num_clusters: int
) -> np.ndarray:
  """Appends one-hot cluster ID to features for VAD models."""
  if not num_clusters:
    return features
  cluster_ids = np.zeros((features.shape[0], num_clusters))
  cluster_ids[:, cluster_id] = 1
  return np.concatenate([features, cluster_ids], axis=1).astype(np.float32)


def normalize_features(
    features: np.ndarray, mean_stddev: Tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
  """Normalizes features with per-bin mean and standard deviation."""
  vad_mean, vad_std_dev = mean_stddev
  mean_vec = np.expand_dims(np.repeat(vad_mean, 4), axis=0)
  std_dev_vec = np.expand_dims(np.repeat(vad_std_dev, 4), axis=0)
  features = (features - mean_vec) / std_dev_vec
  return features.astype(np.float32)


@colortimelog.timefunc
def load_tflite_model(model_path: str) -> tf.lite.Interpreter:
  """Reads a serialized TFLite model and returns an allocated Interpreter."""
  with open(model_path, "rb") as file_object:
    model_content = file_object.read()
  tflite_model = tf.lite.Interpreter(model_content=model_content)
  tflite_model.allocate_tensors()
  tflite_model.reset_all_variables()
  return tflite_model


def run_single_input_model_one_step(
    model: tf.lite.Interpreter, input_data: np.ndarray
) -> np.ndarray:
  """Runs given TFLite model on single input data for a single step."""
  model.set_tensor(
      model.get_input_details()[0]["index"],
      input_data,
  )
  model.invoke()
  return model.get_tensor(model.get_output_details()[0]["index"])


def print_model_info(model: tf.lite.Interpreter) -> None:
  """Prints input and output tensor details."""
  model.reset_all_variables()
  input_details = model.get_input_details()
  output_details = model.get_output_details()
  print("input_details:", input_details)
  print("output_details:", output_details)


def apply_vad(
    features: np.ndarray,
    vad_model: tf.lite.Interpreter,
    mean_stddev: Tuple[np.ndarray, np.ndarray],
    threshold: float,
    cluster_id: int = 2,
    num_clusters: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
  """Applies VAD model inference to audio features."""
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
    model: tf.lite.Interpreter,
    input_data: np.ndarray,
    states: list,
) -> Tuple[np.ndarray, list]:
  """Runs a multi-input recurrent/streaming TFLite model for a single step."""
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
  """Runs given TFLite model on sequential features with state passing."""
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
    features_t = features[start_index:start_index + batch_size, :]
    output, states = run_multi_input_model_one_step(model, features_t, states)
    outputs.append(output)
  return np.concatenate(outputs, axis=0)

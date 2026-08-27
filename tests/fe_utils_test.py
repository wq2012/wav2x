"""Unit tests for fe_utils."""

import os
import unittest
import numpy as np
from wav2x import fe_utils


class FeUtilsTest(unittest.TestCase):

  def test_feature_extractor_default_shape(self):
    """Verifies default LogMelFeatureExtractor output shape."""
    extractor = fe_utils.LogMelFeatureExtractor()
    # 1 second of audio at 16kHz
    input_signal = np.random.uniform(-32768, 32767, (1, 16000)).astype(
        np.int16
    )
    features = extractor.extract(input_signal)

    # Batch size = 1
    self.assertEqual(features.shape[0], 1)
    # Stacking: 128 bins * 4 frames (3 left + 1 current) = 512
    self.assertEqual(features.shape[2], 512)
    # Exactly 33 frames for 16000 samples with 513 window and 160 step
    self.assertEqual(features.shape[1], 33)

  def test_feature_extractor_custom_config(self):
    """Verifies custom configuration of LogMelFeatureExtractor."""
    extractor = fe_utils.LogMelFeatureExtractor(
        num_bins=80,
        stack_left_context=0,
        stack_right_context=0,
        frame_stride=1,
        preemph=0.0,
    )
    input_signal = np.random.uniform(-32768, 32767, (1, 16000)).astype(
        np.float32
    )
    features = extractor.extract(input_signal)
    self.assertEqual(features.shape[0], 1)
    self.assertEqual(features.shape[2], 80)

  def test_feature_extractor_noise(self):
    """Verifies feature extractor runs with dithering noise."""
    extractor = fe_utils.LogMelFeatureExtractor(noise_scale=8.0)
    input_signal = np.zeros((1, 16000), dtype=np.int16)
    features = extractor.extract(input_signal)
    self.assertEqual(features.shape, (1, 33, 512))

  def test_get_int_samples(self):
    """Verifies get_int_samples loads and formats all test WAV files."""
    import glob
    test_wavs = glob.glob("testdata/*.wav")
    if not test_wavs:
      self.skipTest("No test WAV files found in testdata/")
    for wav_path in test_wavs:
      samples = fe_utils.get_int_samples(wav_path)
      self.assertEqual(samples.ndim, 2, f"Failed for {wav_path}")
      self.assertEqual(samples.shape[0], 1, f"Failed for {wav_path}")
      self.assertEqual(samples.dtype, np.int16, f"Failed for {wav_path}")
      self.assertGreater(samples.shape[1], 0, f"Empty audio in {wav_path}")

  def test_add_cluster_id(self):
    """Verifies add_cluster_id appends one-hot cluster representation."""
    features = np.zeros((5, 10), dtype=np.float32)
    augmented = fe_utils.add_cluster_id(
        features, cluster_id=2, num_clusters=16
    )
    self.assertEqual(augmented.shape, (5, 26))
    self.assertEqual(augmented[0, 12], 1.0)
    self.assertEqual(augmented[0, 10], 0.0)

    # When num_clusters is 0
    unchanged = fe_utils.add_cluster_id(features, cluster_id=0, num_clusters=0)
    self.assertEqual(unchanged.shape, (5, 10))

  def test_normalize_features(self):
    """Verifies normalize_features calculates z-score correctly."""
    features = np.ones((2, 8), dtype=np.float32)
    mean = np.zeros(2, dtype=np.float32)
    stddev = np.ones(2, dtype=np.float32) * 2.0
    normalized = fe_utils.normalize_features(features, (mean, stddev))
    self.assertEqual(normalized.shape, (2, 8))
    np.testing.assert_allclose(normalized, 0.5)

  def test_read_mean_stddev_csv(self):
    """Verifies read_mean_stddev_csv parses mean and stddev."""
    csv_file = "models/vad_short_mean_stddev.csv"
    if not os.path.exists(csv_file):
      self.skipTest(f"CSV file not found: {csv_file}")
    mean, stddev = fe_utils.read_mean_stddev_csv(csv_file)
    self.assertIsInstance(mean, np.ndarray)
    self.assertIsInstance(stddev, np.ndarray)
    self.assertEqual(len(mean), len(stddev))


if __name__ == "__main__":
  unittest.main()

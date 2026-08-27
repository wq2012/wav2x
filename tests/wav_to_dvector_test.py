"""Unit tests for wav_to_dvector."""

import unittest
import numpy as np
from wav2x import wav_to_dvector


class WavToDvectorTest(unittest.TestCase):

  def test_compute_cosine(self):
    """Verifies cosine similarity computation."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])
    v4 = np.array([-1.0, 0.0, 0.0])

    self.assertAlmostEqual(wav_to_dvector.compute_cosine(v1, v2), 1.0)
    self.assertAlmostEqual(wav_to_dvector.compute_cosine(v1, v3), 0.0)
    self.assertAlmostEqual(wav_to_dvector.compute_cosine(v1, v4), -1.0)

    # Zero vector handling
    v_zero = np.array([0.0, 0.0, 0.0])
    self.assertEqual(wav_to_dvector.compute_cosine(v1, v_zero), 0.0)

  def test_aggregate_dvectors(self):
    """Verifies d-vector aggregation."""
    dv1 = np.array([1.0, 0.0, 0.0])
    dv2 = np.array([0.0, 1.0, 0.0])
    agg = wav_to_dvector.aggregate_dvectors([dv1, dv2])
    np.testing.assert_allclose(agg, [0.5, 0.5, 0.0])

  def test_runner_init_empty(self):
    """Verifies runner instantiates gracefully without model files."""
    runner = wav_to_dvector.WavToDvectorRunner()
    self.assertIsNone(runner.vad_model)
    self.assertIsNone(runner.tisid_model)
    self.assertIsNone(runner.vad_mean_stddev)

  def test_speaker_enrollment_mock(self):
    """Verifies enrollment and identification logic using runner state."""
    runner = wav_to_dvector.WavToDvectorRunner()
    spk1_vec = np.array([1.0, 0.0, 0.0])
    spk2_vec = np.array([0.0, 1.0, 0.0])

    runner.enrolled_speakers["Alice"] = spk1_vec
    runner.enrolled_speakers["Bob"] = spk2_vec

    # Directly mock wav_to_dvector to return query embedding
    runner.wav_to_dvector = lambda path: np.array([[1.0, 0.0, 0.0]])
    name, score = runner.identify_speaker("dummy.wav", threshold=0.7)
    self.assertEqual(name, "Alice")
    self.assertAlmostEqual(score, 1.0)

    # Test below threshold
    name_low, score_low = runner.identify_speaker("dummy.wav", threshold=1.5)
    self.assertEqual(name_low, "")

    # Clear enrollment
    runner.clear_enrollment()
    self.assertEqual(len(runner.enrolled_speakers), 0)


if __name__ == "__main__":
  unittest.main()

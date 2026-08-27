"""Unit tests for wav_to_lang."""

import unittest
import numpy as np
from wav2x import wav_to_lang


class WavToLangTest(unittest.TestCase):

  def test_runner_init_empty(self):
    """Verifies runner instantiates gracefully without model files."""
    runner = wav_to_lang.WavToLangRunner()
    self.assertIsNone(runner.vad_model)
    self.assertIsNone(runner.langid_model)
    self.assertIsNone(runner.vad_mean_stddev)

  def test_uninitialized_models_raise_error(self):
    """Verifies calling inference without models raises ValueError."""
    runner = wav_to_lang.WavToLangRunner()
    dummy_signal = np.zeros((1, 16000), dtype=np.int16)
    with self.assertRaises(ValueError):
      runner.samples_to_lang(dummy_signal)


if __name__ == "__main__":
  unittest.main()

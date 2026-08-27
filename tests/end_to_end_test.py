"""End-to-end tests calling actual TFLite models."""

import os
import unittest
from wav2x import wav_to_dvector
from wav2x import wav_to_lang


class EndToEndTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.models_dir = "models"
    cls.test_files = {
        "Puck-Arabic": "testdata/Puck-Arabic.wav",
        "Puck-Chinese": "testdata/Puck-Chinese.wav",
        "Puck-English": "testdata/Puck-English.wav",
        "Puck-French": "testdata/Puck-French.wav",
        "Puck-Russian": "testdata/Puck-Russian.wav",
        "Zephyr-Arabic": "testdata/Zephyr-Arabic.wav",
        "Zephyr-Chinese": "testdata/Zephyr-Chinese.wav",
        "Zephyr-English": "testdata/Zephyr-English.wav",
        "Zephyr-French": "testdata/Zephyr-French.wav",
        "Zephyr-Russian": "testdata/Zephyr-Russian.wav",
    }

    # Ensure model files and testdata exist
    needed_models = [
        os.path.join(cls.models_dir, "vad_short_model.tflite"),
        os.path.join(cls.models_dir, "vad_short_mean_stddev.csv"),
        os.path.join(cls.models_dir, "conformer_langid_medium.tflite"),
        os.path.join(cls.models_dir, "vad_long_model.tflite"),
        os.path.join(cls.models_dir, "vad_long_mean_stddev.csv"),
        os.path.join(cls.models_dir, "conformer_tisid_medium.tflite"),
    ]
    cls.models_available = all(os.path.exists(f) for f in needed_models)
    cls.testdata_available = all(
        os.path.exists(f) for f in cls.test_files.values()
    )

    if cls.models_available:
      cls.lang_runner = wav_to_lang.WavToLangRunner.from_pretrained(
          model_dir=cls.models_dir, download_if_missing=False
      )
      cls.sid_runner = wav_to_dvector.WavToDvectorRunner.from_pretrained(
          model_dir=cls.models_dir, download_if_missing=False
      )

  def setUp(self):
    if not self.models_available or not self.testdata_available:
      self.skipTest("Required models or test data not available.")

  def test_language_identification_e2e(self):
    """Verifies end-to-end language identification on all 10 audio files."""
    expected_langs = {
        "Puck-Arabic": "ar-eg",
        "Puck-Chinese": "cmn-hans-cn",
        "Puck-English": "en-us",
        "Puck-French": "fr-fr",
        "Puck-Russian": "ru-ru",
        "Zephyr-Arabic": "ar-eg",
        "Zephyr-Chinese": "cmn-hans-cn",
        "Zephyr-English": "en-us",
        "Zephyr-French": "fr-fr",
        "Zephyr-Russian": "ru-ru",
    }

    for name, path in self.test_files.items():
      lang, probs = self.lang_runner.wav_to_lang(path)
      self.assertEqual(
          lang, expected_langs[name], f"Wrong language predicted for {name}."
      )
      self.assertGreater(
          float(probs.max()),
          0.90,
          f"Confidence too low for {name}.",
      )

  def test_speaker_verification_e2e(self):
    """Verifies end-to-end speaker verification cosine similarity."""
    same_speaker_pairs = [
        ("Puck-Chinese", "Puck-English"),
        ("Puck-Arabic", "Puck-Russian"),
        ("Zephyr-Chinese", "Zephyr-English"),
        ("Zephyr-Arabic", "Zephyr-Russian"),
        ("Zephyr-Arabic", "Zephyr-English"),
    ]
    diff_speaker_pairs = [
        ("Puck-Chinese", "Zephyr-Chinese"),
        ("Puck-English", "Zephyr-English"),
        ("Puck-Chinese", "Zephyr-English"),
        ("Puck-English", "Zephyr-Chinese"),
        ("Puck-Russian", "Zephyr-Russian"),
    ]

    threshold = 0.49

    for name1, name2 in same_speaker_pairs:
      score = self.sid_runner.compute_score(
          [self.test_files[name1]], self.test_files[name2]
      )
      self.assertGreater(
          score,
          threshold,
          f"Score {score} too low for same speaker pair {name1} vs {name2}.",
      )

    for name1, name2 in diff_speaker_pairs:
      score = self.sid_runner.compute_score(
          [self.test_files[name1]], self.test_files[name2]
      )
      self.assertLess(
          score,
          threshold,
          f"Score {score} too high for different speaker {name1} vs {name2}.",
      )

  def test_speaker_identification_e2e(self):
    """Verifies speaker enrollment and identification cross-lingually."""
    # Enroll Puck and Zephyr using Chinese
    self.sid_runner.clear_enrollment()
    self.sid_runner.enroll_speaker("Puck", [self.test_files["Puck-Chinese"]])
    self.sid_runner.enroll_speaker(
        "Zephyr", [self.test_files["Zephyr-Chinese"]]
    )

    threshold = 0.40

    # Identify all remaining 8 audio files across 4 languages
    remaining_files = [
        ("Puck-Arabic", "Puck"),
        ("Puck-English", "Puck"),
        ("Puck-French", "Puck"),
        ("Puck-Russian", "Puck"),
        ("Zephyr-Arabic", "Zephyr"),
        ("Zephyr-English", "Zephyr"),
        ("Zephyr-French", "Zephyr"),
        ("Zephyr-Russian", "Zephyr"),
    ]

    for filename, expected_speaker in remaining_files:
      spk_name, score = self.sid_runner.identify_speaker(
          self.test_files[filename], threshold=threshold
      )
      self.assertEqual(
          spk_name,
          expected_speaker,
          f"Wrong speaker identified for {filename}: {spk_name} vs"
          f" {expected_speaker} (score: {score})",
      )
      self.assertGreater(score, threshold)

    # Multi-utterance enrollment improves embedding quality
    self.sid_runner.clear_enrollment()
    self.sid_runner.enroll_speaker(
        "Puck",
        [self.test_files["Puck-Chinese"], self.test_files["Puck-English"]],
    )
    self.sid_runner.enroll_speaker(
        "Zephyr",
        [self.test_files["Zephyr-Chinese"], self.test_files["Zephyr-English"]],
    )

    multi_test_files = [
        ("Puck-Arabic", "Puck"),
        ("Puck-French", "Puck"),
        ("Puck-Russian", "Puck"),
        ("Zephyr-Arabic", "Zephyr"),
        ("Zephyr-French", "Zephyr"),
        ("Zephyr-Russian", "Zephyr"),
    ]
    for filename, expected_speaker in multi_test_files:
      spk_name, score = self.sid_runner.identify_speaker(
          self.test_files[filename], threshold=0.50
      )
      self.assertEqual(spk_name, expected_speaker)
      self.assertGreater(score, 0.50)

    # High threshold should reject identification
    rejected_name, _ = self.sid_runner.identify_speaker(
        self.test_files["Puck-English"], threshold=0.99
    )
    self.assertEqual(rejected_name, "")


if __name__ == "__main__":
  unittest.main()

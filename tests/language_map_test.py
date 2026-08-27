"""Unit tests for language_map."""

import unittest
from wav2x import language_map


class LanguageMapTest(unittest.TestCase):

  def test_bidirectional_mapping(self):
    """Verifies that LANG_TO_ID and ID_TO_LANG are consistent."""
    for lang, idx in language_map.LANG_TO_ID.items():
      self.assertEqual(language_map.ID_TO_LANG[idx], lang)

    for idx, lang in language_map.ID_TO_LANG.items():
      self.assertEqual(language_map.LANG_TO_ID[lang], idx)

  def test_common_languages_present(self):
    """Verifies that major common languages are defined."""
    common_langs = [
        "en-us",
        "cmn-hans-cn",
        "es-us",
        "fr-fr",
        "de-de",
        "ja-jp",
        "ko-kr",
        "ru-ru",
    ]
    for lang in common_langs:
      self.assertIn(lang, language_map.LANG_TO_ID)


if __name__ == "__main__":
  unittest.main()

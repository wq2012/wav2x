
import numpy as np
import tensorflow.compat.v2 as tf
import unittest
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

# Mock colortimelog if missing, just in case
try:
    import colortimelog
except ImportError:
    import types
    mod = types.ModuleType("colortimelog")
    def timefunc(f): return f
    mod.timefunc = timefunc
    sys.modules["colortimelog"] = mod

import fe_utils
import wav_to_dvector
import wav_to_lang

class TestReplacement(unittest.TestCase):
    def test_feature_extractor_shape(self):
        print("Testing Feature Extractor (Default)...")
        extractor = fe_utils.LogMelFeatureExtractor()
        # 1 second of audio at 16k
        input_signal = np.random.uniform(-1.0, 1.0, (1, 16000)).astype(np.float32)
        # Convert to int16 range as expected by some internals if they did conversion? 
        # But extract takes float or int.
        
        # fe_utils.get_int_samples returns int16 with shape [1, time]
        # Let's test with int16
        input_signal_int = (input_signal * 32768).astype(np.int16)
        
        features = extractor.extract(input_signal_int)
        print(f"Features shape: {features.shape}")
        
        # Checks:
        # Batch size = 1
        self.assertEqual(features.shape[0], 1)
        # Bins = 128 * 4 (stacked) = 512
        self.assertEqual(features.shape[2], 512)
        # Time: ~33 frames for 1 second?
        self.assertTrue(features.shape[1] > 30)
        self.assertTrue(features.shape[1] < 35)

    def test_feature_extractor_config(self):
        print("Testing Feature Extractor (Custom Config)...")
        # Test with 80 bins instead of 128
        extractor = fe_utils.LogMelFeatureExtractor(num_bins=80, stack_left_context=0, stack_right_context=0)
        
        input_signal = np.random.uniform(-1.0, 1.0, (1, 16000)).astype(np.float32)
        input_signal_int = (input_signal * 32768).astype(np.int16)
        
        features = extractor.extract(input_signal_int)
        print(f"Config features shape: {features.shape}")
        
        # Check num bins = 80 * 1 (no stacking) = 80
        self.assertEqual(features.shape[2], 80)

    def test_instantiation(self):
        print("Testing Class Instantiation...")
        runner_dv = wav_to_dvector.WavToDvectorRunner()
        self.assertIsInstance(runner_dv.feature_extractor, fe_utils.LogMelFeatureExtractor)
        
        runner_lang = wav_to_lang.WavToLangRunner()
        self.assertIsInstance(runner_lang.feature_extractor, fe_utils.LogMelFeatureExtractor)

if __name__ == "__main__":
    unittest.main()

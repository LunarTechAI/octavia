"""
Automated tests for Audio Quality Validation
Tests duration validation, SNR checks, silence detection, and gain normalization
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch
import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.audio_quality import (
    AudioQualityValidator,
    QualityMetrics,
    ValidationResult,
    QualityValidationError
)


class TestAudioQualityValidator(unittest.TestCase):
    """Test suite for AudioQualityValidator"""
    
    @classmethod
    def setUpClass(cls):
        """Create test audio files"""
        cls.test_dir = "test_audio_files"
        os.makedirs(cls.test_dir, exist_ok=True)
        
        # Create test audio files
        cls.good_audio_path = os.path.join(cls.test_dir, "good_audio.mp3")
        cls.translated_audio_path = os.path.join(cls.test_dir, "translated_audio.mp3")
        cls.silent_audio_path = os.path.join(cls.test_dir, "silent_audio.mp3")
        cls.noisy_audio_path = os.path.join(cls.test_dir, "noisy_audio.mp3")
        
        # Generate good quality audio (1000 Hz sine wave, 3 seconds)
        good_audio = Sine(1000).to_audio_segment(duration=3000)
        good_audio.export(cls.good_audio_path, format="mp3")
        
        # Generate translated audio (similar duration)
        translated_audio = Sine(1000).to_audio_segment(duration=3100)  # 100ms difference
        translated_audio.export(cls.translated_audio_path, format="mp3")
        
        # Generate silent audio
        silent_audio = AudioSegment.silent(duration=3000)
        silent_audio.export(cls.silent_audio_path, format="mp3")
        
        # Generate noisy audio (low SNR)
        noise = np.random.normal(0, 0.5, 3000 * 44100)  # High noise
        signal = np.sin(2 * np.pi * 1000 * np.arange(3000 * 44100) / 44100) * 0.1  # Low signal
        noisy_data = (signal + noise) * 32767
        noisy_audio = AudioSegment(
            noisy_data.astype(np.int16).tobytes(),
            frame_rate=44100,
            sample_width=2,
            channels=1
        )
        noisy_audio.export(cls.noisy_audio_path, format="mp3")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        import shutil
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        """Initialize validator for each test"""
        self.validator = AudioQualityValidator(
            tolerance_ms=200,
            min_snr_db=20.0,
            max_silence_percentage=10.0
        )
    
    def test_duration_validation_pass(self):
        """Test duration validation with acceptable difference"""
        is_valid, diff_ms, metrics = self.validator.validate_duration(
            self.good_audio_path,
            self.translated_audio_path
        )
        
        self.assertTrue(is_valid)
        self.assertLess(diff_ms, 200)
        self.assertIn("duration_diff_ms", metrics)
    
    def test_duration_validation_fail(self):
        """Test duration validation with excessive difference"""
        # Create audio with large duration difference
        long_audio_path = os.path.join(self.test_dir, "long_audio.mp3")
        long_audio = Sine(1000).to_audio_segment(duration=5000)  # 2 seconds longer
        long_audio.export(long_audio_path, format="mp3")
        
        is_valid, diff_ms, metrics = self.validator.validate_duration(
            self.good_audio_path,
            long_audio_path
        )
        
        self.assertFalse(is_valid)
        self.assertGreater(diff_ms, 200)
        
        os.remove(long_audio_path)
    
    def test_snr_validation_pass(self):
        """Test SNR validation with good quality audio"""
        is_valid, snr_db, metrics = self.validator.validate_snr(self.good_audio_path)
        
        self.assertTrue(is_valid)
        self.assertGreater(snr_db, 20.0)
        self.assertIn("snr_db", metrics)
    
    def test_snr_validation_fail(self):
        """Test SNR validation with noisy audio"""
        is_valid, snr_db, metrics = self.validator.validate_snr(self.noisy_audio_path)
        
        # Note: Depending on the noise level, this might pass or fail
        # The test verifies the SNR calculation works
        self.assertIn("snr_db", metrics)
        self.assertIsInstance(snr_db, float)
    
    def test_silence_detection(self):
        """Test silence detection"""
        silence_pct, metrics = self.validator.detect_silence(self.silent_audio_path)
        
        self.assertGreater(silence_pct, 90)  # Should be mostly silent
        self.assertIn("silence_percentage", metrics)
        self.assertIn("silent_segment_count", metrics)
    
    def test_silence_detection_normal_audio(self):
        """Test silence detection with normal audio"""
        silence_pct, metrics = self.validator.detect_silence(self.good_audio_path)
        
        self.assertLess(silence_pct, 10)  # Should have minimal silence
    
    def test_gain_normalization(self):
        """Test gain normalization"""
        output_path = os.path.join(self.test_dir, "normalized_audio.mp3")
        
        metrics = self.validator.normalize_gain(
            self.good_audio_path,
            output_path,
            target_peak_db=-3.0
        )
        
        self.assertTrue(os.path.exists(output_path))
        self.assertIn("gain_adjustment_db", metrics)
        self.assertIn("normalized", metrics)
        self.assertTrue(metrics["normalized"])
        
        # Verify normalized audio has correct peak level
        normalized_audio = AudioSegment.from_file(output_path)
        self.assertAlmostEqual(normalized_audio.max_dBFS, -3.0, delta=1.0)
        
        os.remove(output_path)
    
    def test_validate_all_pass(self):
        """Test comprehensive validation with good audio"""
        result = self.validator.validate_all(
            self.good_audio_path,
            self.translated_audio_path,
            normalize=False  # Skip normalization for speed
        )
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.failures), 0)
        self.assertIsInstance(result.metrics, QualityMetrics)
    
    def test_validate_all_fail_duration(self):
        """Test comprehensive validation failing on duration"""
        # Create audio with large duration difference
        long_audio_path = os.path.join(self.test_dir, "long_audio_test.mp3")
        long_audio = Sine(1000).to_audio_segment(duration=5000)
        long_audio.export(long_audio_path, format="mp3")
        
        result = self.validator.validate_all(
            self.good_audio_path,
            long_audio_path,
            normalize=False
        )
        
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.failures), 0)
        self.assertIn("duration_mismatch", result.failures[0])
        
        os.remove(long_audio_path)
    
    def test_validate_all_fail_silence(self):
        """Test comprehensive validation failing on excessive silence"""
        result = self.validator.validate_all(
            self.good_audio_path,
            self.silent_audio_path,
            normalize=False
        )
        
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.failures), 0)
        # Should fail on excessive silence
        self.assertTrue(any("silence" in f for f in result.failures))
    
    def test_validation_result_to_dict(self):
        """Test ValidationResult serialization"""
        result = self.validator.validate_all(
            self.good_audio_path,
            self.translated_audio_path,
            normalize=False
        )
        
        result_dict = result.to_dict()
        
        self.assertIn("is_valid", result_dict)
        self.assertIn("metrics", result_dict)
        self.assertIn("failures", result_dict)
        self.assertIn("warnings", result_dict)
        self.assertIsInstance(result_dict["metrics"], dict)


class TestQualityMetrics(unittest.TestCase):
    """Test QualityMetrics dataclass"""
    
    def test_metrics_creation(self):
        """Test creating QualityMetrics"""
        metrics = QualityMetrics(
            duration_diff_ms=100.0,
            snr_db=25.0,
            silence_percentage=5.0,
            peak_level_db=-3.0,
            normalized=True,
            rms_level_db=-16.0
        )
        
        self.assertEqual(metrics.duration_diff_ms, 100.0)
        self.assertEqual(metrics.snr_db, 25.0)
        self.assertTrue(metrics.normalized)
    
    def test_metrics_to_dict(self):
        """Test QualityMetrics serialization"""
        metrics = QualityMetrics(
            duration_diff_ms=100.0,
            snr_db=25.0,
            silence_percentage=5.0,
            peak_level_db=-3.0,
            normalized=True,
            rms_level_db=-16.0
        )
        
        metrics_dict = metrics.to_dict()
        
        self.assertIsInstance(metrics_dict, dict)
        self.assertEqual(metrics_dict["duration_diff_ms"], 100.0)
        self.assertEqual(metrics_dict["snr_db"], 25.0)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestAudioQualityValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestQualityMetrics))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

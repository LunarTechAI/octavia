"""
Audio Quality Validation Module for Octavia
Implements automated quality checks for audio translation jobs:
- Duration validation (±100-200ms tolerance)
- SNR validation (>20dB threshold)
- Silence detection
- Gain normalization
- Metrics storage per job
"""

import os
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_silence
import librosa

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for audio validation"""
    duration_diff_ms: float
    snr_db: float
    silence_percentage: float
    peak_level_db: float
    normalized: bool
    rms_level_db: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ValidationResult:
    """Result of quality validation"""
    is_valid: bool
    metrics: QualityMetrics
    failures: list[str]
    warnings: list[str]
    
    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "metrics": self.metrics.to_dict(),
            "failures": self.failures,
            "warnings": self.warnings
        }


class AudioQualityValidator:
    """
    Validates audio quality for translation jobs
    Ensures output meets quality thresholds
    """
    
    def __init__(
        self,
        tolerance_ms: int = 200,
        min_snr_db: float = 20.0,
        max_silence_percentage: float = 10.0,
        target_peak_db: float = -3.0
    ):
        """
        Initialize validator with quality thresholds
        
        Args:
            tolerance_ms: Duration difference tolerance in milliseconds
            min_snr_db: Minimum acceptable SNR in dB
            max_silence_percentage: Maximum acceptable silence percentage
            target_peak_db: Target peak level for normalization
        """
        self.tolerance_ms = tolerance_ms
        self.min_snr_db = min_snr_db
        self.max_silence_percentage = max_silence_percentage
        self.target_peak_db = target_peak_db
        
        logger.info(f"AudioQualityValidator initialized with thresholds: "
                   f"tolerance={tolerance_ms}ms, min_snr={min_snr_db}dB, "
                   f"max_silence={max_silence_percentage}%")
    
    def validate_duration(
        self,
        original_path: str,
        translated_path: str
    ) -> Tuple[bool, float, Dict]:
        """
        Validate duration difference between original and translated audio
        
        Args:
            original_path: Path to original audio file
            translated_path: Path to translated audio file
            
        Returns:
            Tuple of (is_valid, diff_ms, metrics)
        """
        try:
            original = AudioSegment.from_file(original_path)
            translated = AudioSegment.from_file(translated_path)
            
            original_duration_ms = len(original)
            translated_duration_ms = len(translated)
            diff_ms = abs(translated_duration_ms - original_duration_ms)
            
            is_valid = diff_ms <= self.tolerance_ms
            
            metrics = {
                "original_duration_ms": original_duration_ms,
                "translated_duration_ms": translated_duration_ms,
                "duration_diff_ms": diff_ms,
                "tolerance_ms": self.tolerance_ms
            }
            
            logger.info(f"Duration validation: diff={diff_ms:.2f}ms, "
                       f"valid={is_valid} (tolerance={self.tolerance_ms}ms)")
            
            return is_valid, diff_ms, metrics
            
        except Exception as e:
            logger.error(f"Duration validation failed: {e}")
            raise
    
    def validate_snr(self, audio_path: str) -> Tuple[bool, float, Dict]:
        """
        Validate Signal-to-Noise Ratio of audio
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (is_valid, snr_db, metrics)
        """
        try:
            # Load audio with librosa
            y, sr = librosa.load(audio_path, sr=None)
            
            # Calculate RMS energy
            rms = np.sqrt(np.mean(y**2))
            
            # Estimate noise floor (bottom 10% of energy)
            sorted_energy = np.sort(np.abs(y))
            noise_floor = np.mean(sorted_energy[:int(len(sorted_energy) * 0.1)])
            
            # Calculate SNR in dB
            if noise_floor > 0:
                snr_db = 20 * np.log10(rms / noise_floor)
            else:
                snr_db = 100.0  # Very high SNR if no noise detected
            
            is_valid = snr_db >= self.min_snr_db
            
            metrics = {
                "snr_db": float(snr_db),
                "rms_level": float(rms),
                "noise_floor": float(noise_floor),
                "min_snr_threshold": self.min_snr_db
            }
            
            logger.info(f"SNR validation: snr={snr_db:.2f}dB, "
                       f"valid={is_valid} (threshold={self.min_snr_db}dB)")
            
            return is_valid, snr_db, metrics
            
        except Exception as e:
            logger.error(f"SNR validation failed: {e}")
            raise
    
    def detect_silence(
        self,
        audio_path: str,
        silence_thresh: int = -40
    ) -> Tuple[float, Dict]:
        """
        Detect silent segments in audio
        
        Args:
            audio_path: Path to audio file
            silence_thresh: Silence threshold in dBFS
            
        Returns:
            Tuple of (silence_percentage, metrics)
        """
        try:
            audio = AudioSegment.from_file(audio_path)
            total_duration_ms = len(audio)
            
            # Detect silent segments (min 100ms silence)
            silent_ranges = detect_silence(
                audio,
                min_silence_len=100,
                silence_thresh=silence_thresh
            )
            
            # Calculate total silence duration
            total_silence_ms = sum(end - start for start, end in silent_ranges)
            silence_percentage = (total_silence_ms / total_duration_ms) * 100 if total_duration_ms > 0 else 0
            
            metrics = {
                "silence_percentage": float(silence_percentage),
                "total_silence_ms": float(total_silence_ms),
                "total_duration_ms": float(total_duration_ms),
                "silent_segment_count": len(silent_ranges),
                "silence_threshold_dbfs": silence_thresh
            }
            
            logger.info(f"Silence detection: {silence_percentage:.2f}% silence, "
                       f"{len(silent_ranges)} segments")
            
            return silence_percentage, metrics
            
        except Exception as e:
            logger.error(f"Silence detection failed: {e}")
            raise
    
    def normalize_gain(
        self,
        audio_path: str,
        output_path: str,
        target_peak_db: Optional[float] = None
    ) -> Dict:
        """
        Normalize audio gain to target peak level
        
        Args:
            audio_path: Path to input audio file
            output_path: Path to save normalized audio
            target_peak_db: Target peak level (defaults to instance setting)
            
        Returns:
            Metrics dictionary
        """
        try:
            if target_peak_db is None:
                target_peak_db = self.target_peak_db
            
            audio = AudioSegment.from_file(audio_path)
            
            # Get current peak
            current_peak_db = audio.max_dBFS
            
            # Calculate required gain adjustment
            gain_adjustment_db = target_peak_db - current_peak_db
            
            # Apply gain
            normalized_audio = audio.apply_gain(gain_adjustment_db)
            
            # Export normalized audio
            normalized_audio.export(output_path, format="mp3")
            
            metrics = {
                "original_peak_db": float(current_peak_db),
                "target_peak_db": float(target_peak_db),
                "gain_adjustment_db": float(gain_adjustment_db),
                "normalized": True
            }
            
            logger.info(f"Gain normalization: {current_peak_db:.2f}dB -> "
                       f"{target_peak_db:.2f}dB (adjustment: {gain_adjustment_db:.2f}dB)")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Gain normalization failed: {e}")
            raise
    
    def validate_all(
        self,
        original_path: str,
        translated_path: str,
        normalize: bool = True
    ) -> ValidationResult:
        """
        Run all quality validation checks
        
        Args:
            original_path: Path to original audio file
            translated_path: Path to translated audio file
            normalize: Whether to normalize gain
            
        Returns:
            ValidationResult with comprehensive quality report
        """
        failures = []
        warnings = []
        
        try:
            # 1. Duration validation
            duration_valid, duration_diff, duration_metrics = self.validate_duration(
                original_path, translated_path
            )
            if not duration_valid:
                failures.append(f"duration_mismatch: {duration_diff:.2f}ms exceeds tolerance")
            
            # 2. SNR validation
            snr_valid, snr_db, snr_metrics = self.validate_snr(translated_path)
            if not snr_valid:
                failures.append(f"snr_too_low: {snr_db:.2f}dB below threshold")
            elif snr_db < self.min_snr_db + 5:
                warnings.append(f"snr_marginal: {snr_db:.2f}dB close to threshold")
            
            # 3. Silence detection
            silence_pct, silence_metrics = self.detect_silence(translated_path)
            if silence_pct > self.max_silence_percentage:
                failures.append(f"excessive_silence: {silence_pct:.2f}% exceeds maximum")
            elif silence_pct > self.max_silence_percentage * 0.8:
                warnings.append(f"high_silence: {silence_pct:.2f}% approaching maximum")
            
            # 4. Gain normalization (optional)
            normalization_metrics = {}
            if normalize:
                normalized_path = translated_path.replace(".mp3", "_normalized.mp3")
                normalization_metrics = self.normalize_gain(
                    translated_path,
                    normalized_path
                )
                # Replace original with normalized
                os.replace(normalized_path, translated_path)
            
            # Get peak level
            audio = AudioSegment.from_file(translated_path)
            peak_level_db = audio.max_dBFS
            rms_level_db = audio.dBFS
            
            # Compile metrics
            metrics = QualityMetrics(
                duration_diff_ms=duration_diff,
                snr_db=snr_db,
                silence_percentage=silence_pct,
                peak_level_db=float(peak_level_db),
                normalized=normalize,
                rms_level_db=float(rms_level_db)
            )
            
            is_valid = len(failures) == 0
            
            result = ValidationResult(
                is_valid=is_valid,
                metrics=metrics,
                failures=failures,
                warnings=warnings
            )
            
            logger.info(f"Quality validation complete: valid={is_valid}, "
                       f"failures={len(failures)}, warnings={len(warnings)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            raise


class QualityValidationError(Exception):
    """Raised when audio quality validation fails"""
    pass

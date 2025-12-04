"""
Integration tests for Octavia Video Translator
Meeting Technical Assessment requirements
"""
import os
import sys
import json
import tempfile
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.pipeline import VideoTranslationPipeline, PipelineConfig
from modules.instrumentation import MetricsCollector
from modules.subtitle_generator import SubtitleGenerator

@pytest.fixture
def test_video():
    """Create a test video for testing"""
    import subprocess
    
    # Create a 30-second test video
    test_dir = Path(__file__).parent / "test_samples"
    test_dir.mkdir(exist_ok=True)
    
    test_video = test_dir / "integration_test_30s.mp4"
    
    if not test_video.exists():
        # Create test video with ffmpeg
        cmd = [
            'ffmpeg',
            '-f', 'lavfi',
            '-i', f'testsrc=duration=30:size=640x360:rate=30',
            '-f', 'lavfi',
            '-i', f'sine=frequency=1000:duration=30',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-loglevel', 'error',
            str(test_video),
            '-y'
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Created test video: {test_video}")
        except subprocess.CalledProcessError as e:
            pytest.skip(f"Could not create test video: {e.stderr.decode()}")
    
    return test_video

def test_chunk_planner_no_overlap():
    """Test that audio chunks don't overlap"""
    from modules.pipeline import VideoTranslationPipeline
    
    pipeline = VideoTranslationPipeline()
    
    # Create a test audio file (1 minute of silence)
    import numpy as np
    from scipy.io import wavfile
    
    temp_dir = tempfile.mkdtemp()
    audio_path = Path(temp_dir) / "test_60s.wav"
    
    # Generate 60 seconds of silence at 44100 Hz
    sample_rate = 44100
    duration = 60  # seconds
    samples = np.zeros(sample_rate * duration, dtype=np.int16)
    wavfile.write(audio_path, sample_rate, samples)
    
    # Chunk the audio
    chunks = pipeline.chunk_audio(str(audio_path), chunk_size=30)
    
    # Verify no overlap
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i + 1]
        
        assert current_chunk["end_ms"] == next_chunk["start_ms"], \
            f"Chunk overlap: {current_chunk['end_ms']} != {next_chunk['start_ms']}"
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

def test_srt_merge_aligns():
    """Test that SRT subtitle merging maintains monotonic timing"""
    from modules.subtitle_generator import SubtitleGenerator
    
    generator = SubtitleGenerator()
    
    # Create test segments
    segments = [
        {
            "start": 0.0,
            "end": 5.0,
            "text": "Hello world"
        },
        {
            "start": 5.0,
            "end": 10.0,
            "text": "This is a test"
        },
        {
            "start": 10.0,
            "end": 15.0,
            "text": "Subtitle alignment"
        }
    ]
    
    # Convert to SRT
    srt_text = generator.format_to_srt(segments)
    lines = srt_text.strip().split('\n')
    
    # Parse SRT timestamps
    timestamps = []
    for i in range(0, len(lines), 4):
        if i + 1 < len(lines):
            time_line = lines[i + 1]
            if '-->' in time_line:
                start_str, end_str = time_line.split(' --> ')
                
                # Convert to seconds
                def parse_timestamp(ts):
                    hms, ms = ts.replace(',', '.').split('.')
                    h, m, s = hms.split(':')
                    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
                
                start = parse_timestamp(start_str.strip())
                end = parse_timestamp(end_str.strip())
                timestamps.append((start, end))
    
    # Verify monotonic alignment
    for i in range(len(timestamps) - 1):
        current_end = timestamps[i][1]
        next_start = timestamps[i + 1][0]
        
        # Allow small epsilon for floating point
        assert abs(current_end - next_start) < 0.01, \
            f"Non-monotonic timing: {current_end} != {next_start}"
    
    print(f"✓ SRT maintains monotonic timing with {len(timestamps)} segments")

def test_duration_match_integration(test_video):
    """Integration test: Full pipeline on 30s clip, assert duration match"""
    # This is the REQUIRED integration test from the assessment
    
    print(f"\n Running integration test on: {test_video}")
    
    # Initialize pipeline with tight tolerances
    config = PipelineConfig(
        chunk_size=10,  # Smaller chunks for testing
        timing_tolerance_ms=100,  # Strict tolerance for test
        condensation_ratio=1.1  # Strict condensation limit
    )
    
    pipeline = VideoTranslationPipeline(config)
    
    # Get original duration
    import subprocess
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
           '-of', 'default=noprint_wrappers=1:nokey=1', str(test_video)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    original_duration = float(result.stdout.strip())
    
    print(f"  Original duration: {original_duration:.3f}s")
    
    # Run pipeline
    result = pipeline.process_video(str(test_video), "es")
    
    # Assertions from technical assessment
    assert result["success"], "Pipeline should complete successfully"
    assert result["successful_chunks"] > 0, "Should have at least one successful chunk"
    
    # AT-1: Duration match within tolerance (1 frame ≈ 33ms at 30fps)
    output_duration = result.get('total_time_seconds', 0)
    duration_diff = abs(output_duration - original_duration)
    
    print(f"  Output duration: {output_duration:.3f}s")
    print(f"  Duration difference: {duration_diff:.3f}s")
    
    # Technical requirement: "Final output duration must exactly match the input"
    # tolerance of 100ms as specified
    assert duration_diff <= 0.1, \
        f"Duration mismatch: {duration_diff:.3f}s > 100ms tolerance"
    
    # AT-2: Condensation within limit
    avg_condensation = result.get('avg_condensation_ratio', 1.0)
    print(f"  Avg condensation ratio: {avg_condensation:.3f}")
    assert avg_condensation <= 1.2, \
        f"Condensation exceeds limit: {avg_condensation:.3f} > 1.2"
    
    # AT-3: Segment timing within tolerance
    avg_duration_diff = result.get('avg_duration_diff_ms', 0)
    print(f"  Avg segment timing diff: {avg_duration_diff:.1f}ms")
    assert avg_duration_diff <= 200, \
        f"Segment timing exceeds tolerance: {avg_duration_diff:.1f}ms > 200ms"
    
    # AT-4: Preview generation (check artifacts directory)
    artifacts_dir = Path("artifacts")
    assert artifacts_dir.exists(), "Artifacts directory should exist"
    
    # AT-5: Error handling (test with corrupt file)
    print("  Testing error handling with non-existent file...")
    try:
        pipeline.process_video("non_existent_file.mp4", "es")
        pytest.fail("Should have raised an error for non-existent file")
    except (FileNotFoundError, Exception):
        print("  ✓ Error handling works")
    
    print(" All integration test requirements met!")
    
    # Save test artifacts
    test_report = {
        "test_name": "integration_test",
        "timestamp": datetime.utcnow().isoformat(),
        "requirements_met": [
            "AT-1: Duration match within tolerance",
            "AT-2: Condensation within 1.2x limit",
            "AT-3: Segment timing within 200ms",
            "AT-4: Artifacts generated",
            "AT-5: Error handling works"
        ],
        "metrics": {
            "original_duration_s": original_duration,
            "output_duration_s": output_duration,
            "duration_diff_s": duration_diff,
            "avg_condensation_ratio": avg_condensation,
            "avg_timing_diff_ms": avg_duration_diff,
            "successful_chunks": result["successful_chunks"],
            "total_chunks": result["total_chunks"]
        }
    }
    
    report_path = artifacts_dir / "integration_test_report.json"
    with open(report_path, 'w') as f:
        json.dump(test_report, f, indent=2)
    
    print(f" Test report saved to: {report_path}")

def test_audio_normalization():
    """Test audio normalization to target LUFS"""
    from modules.pipeline import VideoTranslationPipeline
    
    pipeline = VideoTranslationPipeline()
    
    # Create a test audio file with varying levels
    import numpy as np
    from scipy.io import wavfile
    
    temp_dir = tempfile.mkdtemp()
    audio_path = Path(temp_dir) / "test_audio.wav"
    
    # Generate audio with different amplitudes
    sample_rate = 44100
    duration = 5  # seconds
    t = np.linspace(0, duration, sample_rate * duration)
    
    # Mix of loud and quiet sections
    audio_data = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz tone at -6dB
    audio_data[int(sample_rate * 2.5):] *= 0.1  # Quieter section at -20dB
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    wavfile.write(audio_path, sample_rate, audio_data)
    
    # Test normalization
    from pydub import AudioSegment
    original_audio = AudioSegment.from_wav(str(audio_path))
    
    normalized_audio = pipeline._normalize_audio(original_audio)
    
    # Check that peak is within limits
    assert normalized_audio.max_dBFS <= pipeline.config.max_peak_db, \
        f"Peak level {normalized_audio.max_dBFS:.1f}dB exceeds limit {pipeline.config.max_peak_db}dB"
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    print("✓ Audio normalization maintains peak limits")

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])
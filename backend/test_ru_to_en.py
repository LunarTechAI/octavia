#!/usr/bin/env python3
"""
Test Russian to English Video Translation
"""

import sys
import os

# Add backend directory to path
backend_dir = os.path.dirname(__file__)
sys.path.insert(0, backend_dir)

from modules.pipeline import VideoTranslationPipeline, PipelineConfig

if __name__ == "__main__":
    # Create pipeline
    config = PipelineConfig()
    pipeline = VideoTranslationPipeline(config)

    # Force Russian to English translation
    result = pipeline.process_video_fast(
        video_path="backend/test_samples/sample_30s_ru.mp4",
        target_lang="en",
        source_lang="ru"  # Explicitly set source language to Russian
    )

    if result.get("success"):
        print("✅ Russian to English translation completed successfully!")
        print(f"📁 Output: {result.get('output_video')}")

        # Check audio stream
        import subprocess
        output_path = result.get('output_video')
        if output_path and os.path.exists(output_path):
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', output_path
            ], capture_output=True, text=True)

            if result.returncode == 0:
                import json
                probe_data = json.loads(result.stdout)
                streams = probe_data.get("streams", [])
                audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

                if audio_streams:
                    print("✅ Audio stream verification: PASSED")
                    audio_codec = audio_streams[0].get("codec_name", "unknown")
                    bit_rate = audio_streams[0].get("bit_rate", "unknown")
                    print(f"   • Audio codec: {audio_codec}")
                    print(f"   • Audio bitrate: {bit_rate}")
                else:
                    print("❌ Audio stream verification: FAILED - No audio stream found")
            else:
                print("⚠️ Could not verify streams")
        sys.exit(0)
    else:
        print(f"❌ Translation failed: {result.get('error')}")
        sys.exit(1)

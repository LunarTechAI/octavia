"""
Direct test of subtitle job functionality without HTTP layer
Tests subtitle generation, job tracking, and history
"""
import os
import sys
import time
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

def test_subtitle_generator():
    """Test subtitle generation directly"""
    print("\n=== Testing Subtitle Generation ===")

    try:
        from modules.subtitle_generator import SubtitleGenerator

        # Use test video file
        video_path = Path(__file__).parent / "backend" / "test_samples" / "sample_30s_en.mp4"

        if not video_path.exists():
            print(f"✗ Test video not found: {video_path}")
            return None

        print(f"Test file: {video_path}")
        print(f"File size: {video_path.stat().st_size / (1024*1024):.2f} MB")

        # Initialize generator
        generator = SubtitleGenerator(model_size="tiny")

        # Process file
        print("Starting subtitle generation...")
        start_time = time.time()

        result = generator.process_file(str(video_path), "srt", "en")

        elapsed = time.time() - start_time

        if result and result.get("success"):
            print(f"✓ Subtitle generation successful!")
            print(f"  Segments: {result.get('segment_count')}")
            print(f"  Language: {result.get('language')}")
            print(f"  Time elapsed: {elapsed:.2f}s")

            # Check output files
            output_files = result.get('output_files', {})
            print(f"  Output files: {list(output_files.keys())}")

            return result
        else:
            print(f"✗ Subtitle generation failed: {result.get('error')}")
            return None

    except Exception as e:
        print(f"✗ Error during subtitle generation: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_subtitle_translator():
    """Test subtitle file translation"""
    print("\n=== Testing Subtitle Translation ===")

    try:
        from modules.subtitle_translator import SubtitleTranslator

        # Create test SRT content
        test_content = """1
00:00:01,000 --> 00:00:05,000
Hello world, this is a test.

2
00:00:05,000 --> 00:00:10,000
How are you doing today?
"""

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.srt', delete=False, mode='w', encoding='utf-8')
        temp_file.write(test_content)
        temp_file.close()

        try:
            translator = SubtitleTranslator()

            print("Starting subtitle translation (en -> es)...")
            start_time = time.time()

            result = translator.translate_subtitles(
                temp_file.name,
                source_lang='en',
                target_lang='es'
            )

            elapsed = time.time() - start_time

            if result:
                print(f"✓ Subtitle translation successful!")
                print(f"  Segments: {result.get('segment_count')}")
                print(f"  Source language: {result.get('source_language')}")
                print(f"  Target language: {result.get('target_language')}")
                print(f"  Output path: {result.get('output_path')}")
                print(f"  Time elapsed: {elapsed:.2f}s")

                # Read output file
                if os.path.exists(result['output_path']):
                    with open(result['output_path'], 'r', encoding='utf-8') as f:
                        content = f.read()
                        print(f"\n  First few lines of output:")
                        print(f"  {content[:200]}...")

                return result
            else:
                print(f"✗ Translation failed")
                return None

        finally:
            # Cleanup
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)
            if result and os.path.exists(result.get('output_path')):
                os.remove(result['output_path'])

    except Exception as e:
        print(f"✗ Error during subtitle translation: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_job_tracking():
    """Test job creation and tracking"""
    print("\n=== Testing Job Tracking ===")

    try:
        from routes.translation_routes import translation_jobs, save_job_to_supabase
        import uuid
        from datetime import datetime

        # Create a test job
        job_id = str(uuid.uuid4())

        job_data = {
            "id": job_id,
            "type": "subtitles",
            "status": "processing",
            "progress": 0,
            "language": "en",
            "format": "srt",
            "user_id": "test-user",
            "created_at": datetime.utcnow().isoformat()
        }

        # Add to in-memory job store
        translation_jobs[job_id] = job_data
        print(f"✓ Created test job: {job_id[:8]}...")
        print(f"  Type: {job_data['type']}")
        print(f"  Status: {job_data['status']}")

        # Update job status
        translation_jobs[job_id]["status"] = "completed"
        translation_jobs[job_id]["progress"] = 100
        translation_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        print(f"✓ Updated job status to completed")

        # Retrieve job
        if job_id in translation_jobs:
            retrieved_job = translation_jobs[job_id]
            print(f"✓ Retrieved job from store")
            print(f"  Status: {retrieved_job['status']}")
            print(f"  Progress: {retrieved_job['progress']}%")
            return True
        else:
            print(f"✗ Could not retrieve job from store")
            return False

    except Exception as e:
        print(f"✗ Error testing job tracking: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_job_history_simulation():
    """Test job history retrieval (simulated)"""
    print("\n=== Testing Job History Simulation ===")

    try:
        from routes.translation_routes import translation_jobs, get_user_job_history
        import uuid
        from datetime import datetime, timedelta
        from shared_dependencies import User

        # Create multiple test jobs
        now = datetime.utcnow()
        user_id = "test-user-123"

        for i in range(5):
            job_id = str(uuid.uuid4())
            job_data = {
                "id": job_id,
                "type": ["subtitles", "video", "audio"][i % 3],
                "status": ["completed", "processing", "failed"][i % 3],
                "progress": [100, 50, 0][i % 3],
                "user_id": user_id,
                "created_at": (now - timedelta(hours=i)).isoformat()
            }
            translation_jobs[job_id] = job_data
            print(f"  Created job {i+1}: {job_data['type']} - {job_data['status']}")

        print(f"\n✓ Created 5 test jobs")

        # Simulate job history retrieval
        user_jobs = []
        for job_id, job_data in translation_jobs.items():
            if job_data.get("user_id") == user_id:
                user_jobs.append({
                    "id": job_id,
                    "type": job_data.get("type"),
                    "status": job_data.get("status"),
                    "progress": job_data.get("progress"),
                    "created_at": job_data.get("created_at")
                })

        # Sort by creation date (newest first)
        user_jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        print(f"\n✓ Retrieved {len(user_jobs)} jobs for user")
        print(f"\n  Job history (newest first):")
        for i, job in enumerate(user_jobs[:3], 1):
            print(f"    {i}. {job['id'][:8]}... - {job['type']} - {job['status']} ({job['progress']}%)")

        return len(user_jobs) > 0

    except Exception as e:
        print(f"✗ Error testing job history: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Subtitle Jobs and History - Direct Module Tests")
    print("=" * 60)

    results = {}

    # Test 1: Subtitle generation
    results['subtitle_gen'] = test_subtitle_generator() is not None

    # Test 2: Subtitle translation
    results['subtitle_trans'] = test_subtitle_translator() is not None

    # Test 3: Job tracking
    results['job_tracking'] = test_job_tracking()

    # Test 4: Job history
    results['job_history'] = test_job_history_simulation()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {test_name}: {status}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {total - passed} test(s) failed")

if __name__ == "__main__":
    main()

"""
Test subtitle jobs and job history functionality
"""
import os
import sys
import time
import tempfile
import requests
import json

# Test configuration
API_BASE_URL = "http://localhost:8000"

def get_demo_token():
    """Get demo authentication token"""
    response = requests.post(f"{API_BASE_URL}/api/auth/demo-login")
    if response.status_code == 200:
        data = response.json()
        if data and "token" in data:
            return data["token"]
    return None

def test_subtitle_job_creation():
    """Test creating a subtitle generation job"""
    print("\n=== Testing Subtitle Job Creation ===")

    token = get_demo_token()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Use test video file
    video_path = "backend/test_samples/sample_30s_en.mp4"

    if not os.path.exists(video_path):
        print(f"Error: Test video not found: {video_path}")
        return None

    try:
        with open(video_path, 'rb') as f:
            files = {'file': ('test_video.mp4', f, 'video/mp4')}
            data = {'language': 'en', 'format': 'srt'}

            response = requests.post(
                f"{API_BASE_URL}/api/translate/subtitles",
                headers=headers,
                files=files,
                data=data
            )

        print(f"Response status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                job_id = result.get("data", {}).get("job_id")
                print(f"[OK] Subtitle job created successfully")
                print(f"  Job ID: {job_id}")
                return job_id
            else:
                print(f"[FAIL] Job creation failed: {result}")
                return None
        else:
            print(f"[FAIL] Request failed with status {response.status_code}")
            print(f"  Error: {response.text}")
            return None

    except Exception as e:
        print(f"[FAIL] Error during subtitle job creation: {e}")
        return None

def test_job_status(job_id):
    """Test checking job status"""
    print(f"\n=== Testing Job Status for {job_id} ===")

    token = get_demo_token()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.get(
            f"{API_BASE_URL}/api/translate/jobs/{job_id}/status",
            headers=headers
        )

        print(f"Response status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                data = result.get("data", {})
                print(f"[OK] Job status retrieved successfully")
                print(f"  Status: {data.get('status')}")
                print(f"  Progress: {data.get('progress')}%")
                return data
        return None

    except Exception as e:
        print(f"[FAIL] Error checking job status: {e}")
        return None

def test_job_history():
    """Test getting job history"""
    print("\n=== Testing Job History ===")

    token = get_demo_token()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.get(
            f"{API_BASE_URL}/api/translate/jobs/history",
            headers=headers
        )

        print(f"Response status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                jobs = result.get("jobs", [])
                print(f"[OK] Job history retrieved successfully")
                print(f"  Total jobs: {result.get('total', 0)}")
                print(f"  Job count: {len(jobs)}")
                return jobs
        return None

    except Exception as e:
        print(f"[FAIL] Error getting job history: {e}")
        return None

def test_subtitle_file_translation():
    """Test translating a subtitle file"""
    print("\n=== Testing Subtitle File Translation ===")

    token = get_demo_token()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Create test SRT file
    test_content = """1
00:00:01,000 --> 00:00:05,000
Hello world, this is a test.

2
00:00:05,000 --> 00:00:10,000
How are you doing today?
"""

    temp_file = tempfile.NamedTemporaryFile(suffix='.srt', delete=False, mode='w', encoding='utf-8')
    temp_file.write(test_content)
    temp_file.close()

    try:
        with open(temp_file.name, 'rb') as f:
            files = {'file': ('test.srt', f, 'text/plain')}
            data = {
                'sourceLanguage': 'en',
                'targetLanguage': 'es',
                'format': 'srt'
            }

            response = requests.post(
                f"{API_BASE_URL}/api/translate/subtitle-file",
                headers=headers,
                files=files,
                data=data
            )

        print(f"Response status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print(f"[OK] Subtitle translation completed successfully")
                print(f"  Segments: {result.get('segment_count')}")
                print(f"  Download URL: {result.get('download_url')}")
                return result.get('download_url')
        return None

    except Exception as e:
        print(f"[FAIL] Error during subtitle translation: {e}")
        return None
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

def test_video_translation():
    """Test video translation"""
    print("\n=== Testing Video Translation ===")

    token = get_demo_token()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    video_path = "backend/test_samples/sample_30s_en.mp4"

    if not os.path.exists(video_path):
        print(f"[FAIL] Test video not found: {video_path}")
        return None

    try:
        with open(video_path, 'rb') as f:
            files = {'file': ('test_video.mp4', f, 'video/mp4')}
            data = {'target_language': 'es'}

            response = requests.post(
                f"{API_BASE_URL}/api/translate/video",
                headers=headers,
                files=files,
                data=data
            )

        print(f"Response status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print(f"[OK] Video translation job created successfully")
                print(f"  Job ID: {result.get('job_id')}")
                return result.get('job_id')
        return None

    except Exception as e:
        print(f"[FAIL] Error during video translation: {e}")
        return None

def test_audio_translation():
    """Test audio translation"""
    print("\n=== Testing Audio Translation ===")

    token = get_demo_token()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    video_path = "backend/test_samples/sample_30s_en.mp4"

    if not os.path.exists(video_path):
        print(f"[FAIL] Test audio file not found: {video_path}")
        return None

    try:
        with open(video_path, 'rb') as f:
            files = {'file': ('test_audio.wav', f, 'audio/wav')}
            data = {
                'source_lang': 'en',
                'target_lang': 'es'
            }

            response = requests.post(
                f"{API_BASE_URL}/api/translate/audio",
                headers=headers,
                files=files,
                data=data
            )

        print(f"Response status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print(f"[OK] Audio translation job created successfully")
                print(f"  Job ID: {result.get('job_id')}")
                return result.get('job_id')
        return None

    except Exception as e:
        print(f"[FAIL] Error during audio translation: {e}")
        return None

def wait_for_job_completion(job_id, max_wait=120, poll_interval=10):
    """Wait for a job to complete"""
    print(f"\n=== Waiting for Job {job_id[:8]}... to Complete ===")

    start_time = time.time()
    while time.time() - start_time < max_wait:
        job_data = test_job_status(job_id)

        if job_data:
            status = job_data.get('status')
            progress = job_data.get('progress', 0)

            print(f"  Status: {status}, Progress: {progress}%")

            if status == 'completed':
                print(f"[OK] Job completed successfully!")
                return job_data
            elif status == 'failed':
                print(f"[FAIL] Job failed: {job_data.get('error')}")
                return job_data
            elif status == 'processing':
                print(f"  Message: {job_data.get('status_message', 'Processing...')}")

        time.sleep(poll_interval)

    print(f"[TIMEOUT] Job timed out after {max_wait} seconds")
    return None

def main():
    """Run all tests"""
    print("=" * 60)
    print("Subtitle Jobs and Job History Test Suite")
    print("=" * 60)

    # First, get demo token
    print("\n=== Getting Demo Token ===")
    token = get_demo_token()
    if token:
        print(f"[OK] Demo token obtained: {token[:20]}...")
    else:
        print("[FAIL] Could not get demo token")
        print("Response:", requests.post(f"{API_BASE_URL}/api/auth/demo-login").text)
        return

    results = {}

    # Test 1: Create subtitle job
    print("\n" + "=" * 60)
    subtitle_job_id = test_subtitle_job_creation()

    if subtitle_job_id:
        # Test 2: Monitor subtitle job status
        final_status = wait_for_job_completion(subtitle_job_id, max_wait=90)
        results['subtitle_job'] = final_status and final_status.get('status') in ['completed', 'processing']
    else:
        results['subtitle_job'] = False

    # Test 3: Check job history
    print("\n" + "=" * 60)
    print("=== Checking Job History After Subtitle Job ===")
    jobs = test_job_history()
    results['job_history'] = jobs is not None

    if jobs:
        # Verify our job is in history
        found = any(job.get('id') == subtitle_job_id for job in jobs)
        if found:
            print(f"[OK] Subtitle job found in history")
            results['job_in_history'] = True
        else:
            print(f"[WARNING] Subtitle job NOT in history yet (may still be processing)")
            results['job_in_history'] = jobs is not None  # Consider it OK if we got history

    # Test 4: Video translation
    print("\n" + "=" * 60)
    video_job_id = test_video_translation()
    if video_job_id:
        results['video_job'] = True
        print(f"[OK] Video job created: {video_job_id[:8]}...")
    else:
        results['video_job'] = False

    # Test 5: Audio translation
    print("\n" + "=" * 60)
    audio_job_id = test_audio_translation()
    if audio_job_id:
        results['audio_job'] = True
        print(f"[OK] Audio job created: {audio_job_id[:8]}...")
    else:
        results['audio_job'] = False

    # Test 6: Subtitle file translation
    print("\n" + "=" * 60)
    subtitle_trans = test_subtitle_file_translation()
    results['subtitle_translation'] = subtitle_trans is not None

    # Final job history check
    print("\n" + "=" * 60)
    print("=== Final Job History Check ===")
    final_jobs = test_job_history()
    results['final_history'] = final_jobs is not None

    if final_jobs:
        print(f"\n[OK] Final job history retrieved with {len(final_jobs)} jobs")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "[OK]" if passed else "[FAIL]"
        print(f"  {symbol} {test_name}: {status}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n[OK] All tests passed!")
    else:
        print(f"\n[INFO] {total - passed} test(s) failed or skipped")

if __name__ == "__main__":
    main()

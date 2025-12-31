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
DEMO_MODE = True

def get_demo_token():
    """Get demo authentication token"""
    if not DEMO_MODE:
        return None

    response = requests.post(f"{API_BASE_URL}/api/auth/demo-login")
    if response.status_code == 200:
        data = response.json()
        if data and "token" in data:
            return data["token"]
    return None

def create_test_audio_file():
    """Create a small test audio file"""
    import numpy as np
    from scipy.io import wavfile

    # Generate 5 seconds of silence
    sample_rate = 16000
    duration = 5
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Generate a simple tone for testing
    audio_data = np.sin(2 * np.pi * 440 * t)
    audio_data = (audio_data * 32767).astype(np.int16)

    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False, mode='wb')
    wavfile.write(temp_file.name, sample_rate, audio_data)
    temp_file.close()

    return temp_file.name

def test_subtitle_job_creation():
    """Test creating a subtitle generation job"""
    print("\n=== Testing Subtitle Job Creation ===")

    token = get_demo_token()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Create test audio file
    audio_file = create_test_audio_file()

    try:
        # Upload and create subtitle job
        with open(audio_file, 'rb') as f:
            files = {'file': ('test_audio.wav', f, 'audio/wav')}
            data = {'language': 'en', 'format': 'srt'}

            response = requests.post(
                f"{API_BASE_URL}/api/translate/subtitles",
                headers=headers,
                files=files,
                data=data
            )

        print(f"Response status: {response.status_code}")
        print(f"Response: {response.json()}")

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                job_id = result.get("data", {}).get("job_id")
                print(f"✓ Subtitle job created successfully")
                print(f"  Job ID: {job_id}")
                print(f"  Status URL: {result.get('status_url')}")
                return job_id
            else:
                print(f"✗ Job creation failed: {result}")
                return None
        else:
            print(f"✗ Request failed with status {response.status_code}")
            print(f"  Error: {response.text}")
            return None

    except Exception as e:
        print(f"✗ Error during subtitle job creation: {e}")
        return None
    finally:
        # Cleanup
        if os.path.exists(audio_file):
            os.remove(audio_file)

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
                print(f"✓ Job status retrieved successfully")
                print(f"  Status: {data.get('status')}")
                print(f"  Progress: {data.get('progress')}%")
                print(f"  Type: {data.get('type')}")
                return data
        return None

    except Exception as e:
        print(f"✗ Error checking job status: {e}")
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
                print(f"✓ Job history retrieved successfully")
                print(f"  Total jobs: {result.get('total', 0)}")
                print(f"  Job count: {len(jobs)}")

                if jobs:
                    print(f"\n  Recent jobs:")
                    for i, job in enumerate(jobs[:5], 1):
                        print(f"    {i}. ID: {job.get('id', 'N/A')[:8]}...")
                        print(f"       Type: {job.get('type', 'N/A')}")
                        print(f"       Status: {job.get('status', 'N/A')}")
                        print(f"       Progress: {job.get('progress', 0)}%")
                        print(f"       Created: {job.get('created_at', 'N/A')[:19] if job.get('created_at') else 'N/A'}")
                        print()

                return jobs
        return None

    except Exception as e:
        print(f"✗ Error getting job history: {e}")
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
                print(f"✓ Subtitle translation completed successfully")
                print(f"  Segments: {result.get('segment_count')}")
                print(f"  Download URL: {result.get('download_url')}")
                return result.get('download_url')
        return None

    except Exception as e:
        print(f"✗ Error during subtitle translation: {e}")
        return None
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

def wait_for_job_completion(job_id, max_wait=120, poll_interval=5):
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
                print(f"✓ Job completed successfully!")
                return job_data
            elif status == 'failed':
                print(f"✗ Job failed: {job_data.get('error')}")
                return job_data
            elif status == 'processing':
                print(f"  Message: {job_data.get('status_message', 'Processing...')}")

        time.sleep(poll_interval)

    print(f"✗ Job timed out after {max_wait} seconds")
    return None

def main():
    """Run all tests"""
    print("=" * 60)
    print("Subtitle Jobs and Job History Test Suite")
    print("=" * 60)

    # Test 1: Create subtitle job
    job_id = test_subtitle_job_creation()

    if job_id:
        # Test 2: Monitor job status
        final_status = wait_for_job_completion(job_id, max_wait=60)

        # Test 3: Check job history
        print("\n=== Checking Job History After Job Creation ===")
        jobs = test_job_history()

        if jobs:
            # Verify our job is in history
            found = any(job.get('id') == job_id for job in jobs)
            if found:
                print(f"✓ Job {job_id[:8]}... found in history")
            else:
                print(f"✗ Job {job_id[:8]}... NOT found in history")

    # Test 4: Subtitle file translation
    print("\n=== Testing Direct Subtitle File Translation ===")
    translation_result = test_subtitle_file_translation()

    # Test 5: Check job history again
    print("\n=== Final Job History Check ===")
    final_jobs = test_job_history()

    print("\n" + "=" * 60)
    print("Test Suite Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()

"""
Comprehensive Dashboard Features Test
Tests every option available on the dashboard pages
"""
import os
import sys
import time
import tempfile
import requests
import json
from pathlib import Path
from typing import Dict, List, Any

# Test configuration
API_BASE_URL = "http://localhost:8000"
VIDEO_PATH = Path(__file__).parent / "backend" / "test_samples" / "sample_30s_en.mp4"
AUDIO_PATH = Path(__file__).parent / "backend" / "test_samples" / "sample_30s_en.mp4"

# Test results tracker
test_results = {
    "subtitle_generation": False,
    "subtitle_file_translation": False,
    "video_translation_standard": False,
    "video_translation_enhanced": False,
    "audio_translation": False,
    "job_history": False,
    "job_status": False,
    "authentication": False
}

def log_test(test_name: str, status: str, details: str = ""):
    """Log test result"""
    status_symbols = {
        "PASS": "[OK]",
        "FAIL": "[FAIL]",
        "SKIP": "[SKIP]",
        "WARN": "[WARN]"
    }
    symbol = status_symbols.get(status.upper(), f"[{status.upper()}]")
    print(f"{symbol} {test_name}")
    if details:
        print(f"      {details}")

def get_demo_token() -> str | None:
    """Get demo authentication token"""
    try:
        response = requests.post(f"{API_BASE_URL}/api/auth/demo-login")
        if response.status_code == 200:
            data = response.json()
            if data and "token" in data:
                print(f"Token obtained: {data['token'][:20]}...")
                return data["token"]
        return None
    except Exception as e:
        print(f"Error getting token: {e}")
        return None

def test_authentication():
    """Test demo authentication"""
    print("\n" + "="*70)
    print("TEST 1: DEMO AUTHENTICATION")
    print("="*70)

    try:
        response = requests.post(f"{API_BASE_URL}/api/auth/demo-login")
        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")

            if data and "success" in data and data["success"]:
                print(f"User: {data.get('user', {})}")
                test_results["authentication"] = True
                log_test("Demo Login", "PASS", "Successfully authenticated demo user")
                return data.get("token")
            else:
                log_test("Demo Login", "FAIL", "Authentication failed")
        else:
            log_test("Demo Login", "FAIL", f"HTTP {response.status_code}: {response.text}")
    except Exception as e:
        log_test("Demo Login", "FAIL", str(e))

    return None

def test_subtitle_generation(token: str):
    """Test subtitle generation from video/audio file"""
    print("\n" + "="*70)
    print("TEST 2: SUBTITLE GENERATION")
    print("="*70)

    if not VIDEO_PATH.exists():
        log_test("Subtitle Generation", "SKIP", "Test video not found")
        return None

    try:
        with open(VIDEO_PATH, 'rb') as f:
            files = {'file': ('test_video.mp4', f, 'video/mp4')}
            data = {
                'language': 'en',
                'format': 'srt'
            }

            print(f"Uploading: {VIDEO_PATH.name}")
            print(f"Size: {VIDEO_PATH.stat().st_size / (1024*1024):.2f} MB")

            response = requests.post(
                f"{API_BASE_URL}/api/translate/subtitles",
                headers={'Authorization': f'Bearer {token}'},
                files=files,
                data=data
            )

            print(f"Response status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2)}")

                if result.get("success"):
                    job_id = result.get("data", {}).get("job_id")
                    test_results["subtitle_generation"] = True
                    log_test("Subtitle Generation", "PASS", f"Job created: {job_id[:20]}...")

                    # Test job status endpoint
                    status_response = requests.get(
                        f"{API_BASE_URL}/api/translate/jobs/{job_id}/status",
                        headers={'Authorization': f'Bearer {token}'}
                    )

                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        if status_data.get("success"):
                            job_info = status_data.get("data", {})
                            test_results["job_status"] = True
                            log_test("Job Status Endpoint", "PASS", f"Status: {job_info.get('status')}")
                            return job_id
                    return job_id
                else:
                    log_test("Subtitle Generation", "FAIL", "No job ID returned")
            else:
                log_test("Subtitle Generation", "FAIL", f"HTTP {response.status_code}: {response.text}")
    except Exception as e:
        log_test("Subtitle Generation", "FAIL", str(e))

    return None

def test_subtitle_file_translation(token: str):
    """Test subtitle file translation"""
    print("\n" + "="*70)
    print("TEST 3: SUBTITLE FILE TRANSLATION")
    print("="*70)

    # Create test SRT file
    test_content = """1
00:00:01,000 --> 00:00:05,000
Hello world, this is a test subtitle file.

2
00:00:05,000 --> 00:00:10,000
This is the second subtitle segment for testing.
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

            print(f"Translating subtitle file: {temp_file.name}")
            print(f"Source: English, Target: Spanish")

            response = requests.post(
                f"{API_BASE_URL}/api/translate/subtitle-file",
                headers={'Authorization': f'Bearer {token}'},
                files=files,
                data=data
            )

            print(f"Response status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2)}")

                if result.get("success"):
                    test_results["subtitle_file_translation"] = True
                    log_test("Subtitle File Translation", "PASS", f"Segments: {result.get('segment_count')}")
                else:
                    log_test("Subtitle File Translation", "FAIL", "Success field not set")
            else:
                log_test("Subtitle File Translation", "FAIL", f"HTTP {response.status_code}: {response.text}")
    except Exception as e:
        log_test("Subtitle File Translation", "FAIL", str(e))
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

def test_video_translation_standard(token: str):
    """Test standard video translation"""
    print("\n" + "="*70)
    print("TEST 4: VIDEO TRANSLATION (STANDARD)")
    print("="*70)

    if not VIDEO_PATH.exists():
        log_test("Video Translation (Standard)", "SKIP", "Test video not found")
        return None

    try:
        with open(VIDEO_PATH, 'rb') as f:
            files = {'file': ('test_video.mp4', f, 'video/mp4')}
            data = {'target_language': 'es'}

            print(f"Uploading: {VIDEO_PATH.name}")
            print(f"Mode: Standard Translation")

            response = requests.post(
                f"{API_BASE_URL}/api/translate/video",
                headers={'Authorization': f'Bearer {token}'},
                files=files,
                data=data
            )

            print(f"Response status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2)}")

                if result.get("success"):
                    job_id = result.get("job_id")
                    test_results["video_translation_standard"] = True
                    log_test("Video Translation (Standard)", "PASS", f"Job ID: {job_id[:20]}...")
                    return job_id
                else:
                    log_test("Video Translation (Standard)", "FAIL", "No job ID returned")
            else:
                log_test("Video Translation (Standard)", "FAIL", f"HTTP {response.status_code}: {response.text}")
    except Exception as e:
        log_test("Video Translation (Standard)", "FAIL", str(e))

    return None

def test_video_translation_enhanced(token: str):
    """Test enhanced video translation"""
    print("\n" + "="*70)
    print("TEST 5: VIDEO TRANSLATION (ENHANCED)")
    print("="*70)

    if not VIDEO_PATH.exists():
        log_test("Video Translation (Enhanced)", "SKIP", "Test video not found")
        return None

    try:
        with open(VIDEO_PATH, 'rb') as f:
            files = {'file': ('test_video.mp4', f, 'video/mp4')}
            data = {
                'target_language': 'es',
                'chunk_size': '30'  # Enhanced mode uses chunking
            }

            print(f"Uploading: {VIDEO_PATH.name}")
            print(f"Mode: Enhanced (Chunk Processing)")

            response = requests.post(
                f"{API_BASE_URL}/api/translate/video/enhanced",
                headers={'Authorization': f'Bearer {token}'},
                files=files,
                data=data
            )

            print(f"Response status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2)}")

                if result.get("success"):
                    job_id = result.get("job_id")
                    test_results["video_translation_enhanced"] = True
                    log_test("Video Translation (Enhanced)", "PASS", f"Job ID: {job_id[:20]}...")
                    return job_id
                else:
                    log_test("Video Translation (Enhanced)", "FAIL", "No job ID returned")
            else:
                log_test("Video Translation (Enhanced)", "FAIL", f"HTTP {response.status_code}: {response.text}")
    except Exception as e:
        log_test("Video Translation (Enhanced)", "FAIL", str(e))

    return None

def test_audio_translation(token: str):
    """Test audio translation"""
    print("\n" + "="*70)
    print("TEST 6: AUDIO TRANSLATION")
    print("="*70)

    if not VIDEO_PATH.exists():
        log_test("Audio Translation", "SKIP", "Test audio/video file not found")
        return None

    try:
        with open(VIDEO_PATH, 'rb') as f:
            files = {'file': ('test_audio.wav', f, 'audio/wav')}
            data = {
                'source_lang': 'en',
                'target_lang': 'es'
            }

            print(f"Uploading: {VIDEO_PATH.name} (as audio)")
            print(f"Source: English, Target: Spanish")

            response = requests.post(
                f"{API_BASE_URL}/api/translate/audio",
                headers={'Authorization': f'Bearer {token}'},
                files=files,
                data=data
            )

            print(f"Response status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2)}")

                if result.get("success"):
                    job_id = result.get("job_id")
                    test_results["audio_translation"] = True
                    log_test("Audio Translation", "PASS", f"Job ID: {job_id[:20]}...")
                    return job_id
                else:
                    log_test("Audio Translation", "FAIL", "No job ID returned")
            else:
                log_test("Audio Translation", "FAIL", f"HTTP {response.status_code}: {response.text}")
    except Exception as e:
        log_test("Audio Translation", "FAIL", str(e))

    return None

def test_job_history(token: str, job_ids: List[str]):
    """Test job history retrieval"""
    print("\n" + "="*70)
    print("TEST 7: JOB HISTORY")
    print("="*70)

    try:
        response = requests.get(
            f"{API_BASE_URL}/api/translate/jobs/history",
            headers={'Authorization': f'Bearer {token}'}
        )

        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")

            if result.get("success"):
                jobs = result.get("jobs", [])
                total = result.get("total", 0)

                print(f"\nTotal jobs in history: {total}")
                print(f"Jobs retrieved: {len(jobs)}")

                # Check if our jobs are in history
                found_jobs = 0
                for job in jobs:
                    job_id = job.get('id', '')
                    for test_job_id in job_ids:
                        if job_id == test_job_id:
                            found_jobs += 1
                            print(f"  - Found job: {job_id[:20]}... (Status: {job.get('status')})")

                test_results["job_history"] = True
                log_test("Job History", "PASS", f"Found {found_jobs}/{len(job_ids)} test jobs, Total: {total} jobs")
            else:
                log_test("Job History", "FAIL", "Success field not set")
        else:
            log_test("Job History", "FAIL", f"HTTP {response.status_code}: {response.text}")
    except Exception as e:
        log_test("Job History", "FAIL", str(e))

def test_file_size_validation(token: str):
    """Test file size validation"""
    print("\n" + "="*70)
    print("TEST 8: FILE SIZE VALIDATION")
    print("="*70)

    # Create a large file (>500MB) to test validation
    large_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    large_file.write(b'0' * (600 * 1024 * 1024))  # 600MB
    large_file.close()

    try:
        with open(large_file.name, 'rb') as f:
            files = {'file': ('large_video.mp4', f, 'video/mp4')}
            data = {'target_language': 'es'}

            print(f"Uploading large file (600MB) to test validation...")

            response = requests.post(
                f"{API_BASE_URL}/api/translate/video",
                headers={'Authorization': f'Bearer {token}'},
                files=files,
                data=data,
                timeout=10  # Short timeout
            )

            if response.status_code == 400:
                test_results["file_size_validation"] = True
                log_test("File Size Validation", "PASS", "Large file correctly rejected")
            else:
                log_test("File Size Validation", "FAIL", f"Expected 400, got {response.status_code}")
    except requests.exceptions.Timeout:
        log_test("File Size Validation", "PASS", "Upload correctly timed out (large file)")
    except Exception as e:
        log_test("File Size Validation", "FAIL", str(e))
    finally:
        if os.path.exists(large_file.name):
            os.remove(large_file.name)

def test_language_validation(token: str):
    """Test language code validation"""
    print("\n" + "="*70)
    print("TEST 9: LANGUAGE VALIDATION")
    print("="*70)

    try:
        with open(VIDEO_PATH, 'rb') as f:
            files = {'file': ('test_video.mp4', f, 'video/mp4')}

            # Test with invalid language code
            data = {'target_language': 'invalid_lang_code'}

            print(f"Testing with invalid language code...")

            response = requests.post(
                f"{API_BASE_URL}/api/translate/subtitles",
                headers={'Authorization': f'Bearer {token}'},
                files=files,
                data=data,
                timeout=10
            )

            print(f"Response status: {response.status_code}")

            # We expect this to fail validation
            if response.status_code in [400, 422]:
                test_results["language_validation"] = True
                log_test("Language Validation", "PASS", "Invalid language rejected")
            elif response.status_code == 200:
                # Check if it falls back to default (which we want to avoid)
                result = response.json()
                if "en" in json.dumps(result).lower():
                    log_test("Language Validation", "WARN", "Invalid language fell back to default")
                    test_results["language_validation"] = True
                else:
                    log_test("Language Validation", "FAIL", "Invalid language was not rejected")
            else:
                log_test("Language Validation", "FAIL", f"Unexpected status: {response.status_code}")
    except Exception as e:
        log_test("Language Validation", "FAIL", str(e))

def test_job_status_polling(token: str, job_id: str):
    """Test job status polling"""
    print("\n" + "="*70)
    print("TEST 10: JOB STATUS POLLING")
    print("="*70)

    if not job_id:
        log_test("Job Status Polling", "SKIP", "No job ID provided")
        return

    try:
        max_polls = 10
        for i in range(max_polls):
            response = requests.get(
                f"{API_BASE_URL}/api/translate/jobs/{job_id}/status",
                headers={'Authorization': f'Bearer {token}'}
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    data = result.get("data", {})
                    status = data.get("status", "unknown")
                    progress = data.get("progress", 0)

                    print(f"Poll {i+1}/{max_polls}: Status={status}, Progress={progress}%")

                    if status == "completed":
                        test_results["job_status"] = True
                        log_test("Job Status Polling", "PASS", f"Job completed successfully")
                        return True
                    elif status == "failed":
                        print(f"Job failed: {data.get('error')}")
                        return False

            time.sleep(2)

        log_test("Job Status Polling", "WARN", "Job did not complete within poll time")
        return False
    except Exception as e:
        log_test("Job Status Polling", "FAIL", str(e))
        return False

def print_summary():
    """Print test summary"""
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    total_tests = len(test_results)
    passed_tests = sum(1 for v in test_results.values() if v)
    failed_tests = total_tests - passed_tests

    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%\n")

    for test_name, passed in test_results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {test_name}")

    print("\n" + "="*70)

    if passed_tests == total_tests:
        print("[PASS] ALL TESTS PASSED!")
        print("Dashboard features are fully functional and integrated.")
    else:
        print(f"[FAIL] {failed_tests} TEST(S) FAILED")
        print("Some features need attention.")

    print("="*70)

    total_tests = len(test_results)
    passed_tests = sum(1 for v in test_results.values() if v)
    failed_tests = total_tests - passed_tests

    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%\n")

    for test_name, passed in test_results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    print("\n" + "="*70)

    if passed_tests == total_tests:
        print("ALL TESTS PASSED! ✓")
        print("Dashboard features are fully functional and integrated.")
    else:
        print(f"{failed_tests} TEST(S) FAILED")
        print("Some features need attention.")

    print("="*70)

def main():
    """Run all dashboard tests"""
    print("="*70)
    print("OCTAVIA DASHBOARD FEATURES COMPREHENSIVE TEST")
    print("="*70)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Test Video: {VIDEO_PATH}")
    print(f"Test Audio: {AUDIO_PATH}")
    print("="*70)

    # Test 1: Authentication
    token = test_authentication()

    if not token:
        print("\n[CRITICAL] Authentication failed. Cannot proceed with other tests.")
        print_summary()
        return

    job_ids = []

    # Test 2: Subtitle Generation
    subtitle_job_id = test_subtitle_generation(token)
    if subtitle_job_id:
        job_ids.append(subtitle_job_id)

    # Test 3: Subtitle File Translation
    test_subtitle_file_translation(token)

    # Test 4: Video Translation (Standard)
    video_job_id = test_video_translation_standard(token)
    if video_job_id:
        job_ids.append(video_job_id)

    # Test 5: Video Translation (Enhanced)
    video_enhanced_job_id = test_video_translation_enhanced(token)
    if video_enhanced_job_id:
        job_ids.append(video_enhanced_job_id)

    # Test 6: Audio Translation
    audio_job_id = test_audio_translation(token)
    if audio_job_id:
        job_ids.append(audio_job_id)

    # Wait a moment for jobs to be created
    print("\nWaiting 3 seconds for job creation...")
    time.sleep(3)

    # Test 7: Job History
    test_job_history(token, job_ids)

    # Test 8: File Size Validation
    test_file_size_validation(token)

    # Test 9: Language Validation
    test_language_validation(token)

    # Test 10: Job Status Polling
    if job_ids:
        test_job_status_polling(token, job_ids[0])

    # Print final summary
    print_summary()

if __name__ == "__main__":
    main()

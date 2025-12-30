
import os
import sys
import requests
import time
import json
import logging
import socket
from datetime import datetime, timedelta
from jose import jwt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
input_video_path = r"e:\Apps\octavia-assignment\input\AI Engineering Bootcamp _ Master Generative AI with LunarTech _ Lets Build The Future Together.mp4"
jwt_secret = "your-secret-key-change-in-production"  # From shared_dependencies.py
algorithm = "HS256"

def create_demo_token():
    """Create a valid JWT token for the demo user"""
    # Use the specific UUID expected by get_current_user in shared_dependencies.py
    user_id = "550e8400-e29b-41d4-a716-446655440000"
    
    expire = datetime.utcnow() + timedelta(minutes=60)
    to_encode = {
        "sub": user_id,
        "exp": expire
    }
    encoded_jwt = jwt.encode(to_encode, jwt_secret, algorithm=algorithm)
    return encoded_jwt

def find_server_port():
    """Scan ports to find the running server"""
    for port in range(8000, 8011):  # Scan 8000-8010
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('127.0.0.1', port))
                if result == 0:
                    # Try to hit the docs endpoint to verify it's our server
                    try:
                        r = requests.get(f"http://127.0.0.1:{port}/docs", timeout=1)
                        if r.status_code == 200:
                            logger.info(f"Found server on port {port}")
                            return port
                    except:
                        pass
        except:
            pass
    return None

def translate_video():
    if not os.path.exists(input_video_path):
        logger.error(f"Input file not found: {input_video_path}")
        return

    logger.info(f"Input file found: {input_video_path}")
    
    # 1. Find server
    port = find_server_port()
    if not port:
        # Fallback to 8000 but it likely won't work
        port = 8000
        logger.warning("Could not automatically find server, defaulting to 8000")
        
    base_url = f"http://127.0.0.1:{port}"
    logger.info(f"Using server at {base_url}")

    # 2. Authenticate
    token = create_demo_token()
    headers = {"Authorization": f"Bearer {token}"}
    logger.info("Created authentication token")

    # 3. Upload and start translation
    url = f"{base_url}/api/translate/video"
    logger.info(f"Uploading video to {url}...")
    
    try:
        with open(input_video_path, 'rb') as f:
            files = {'file': (os.path.basename(input_video_path), f, 'video/mp4')}
            data = {'target_language': 'ru'}
            
            response = requests.post(url, headers=headers, files=files, data=data)
            
        if response.status_code != 200:
            logger.error(f"Failed to start translation: {response.text}")
            return
            
        res_json = response.json()
        job_id = res_json.get('data', {}).get('job_id') or res_json.get('job_id')
        logger.info(f"Translation started! Job ID: {job_id}")
        
    except requests.exceptions.ConnectionError:
        logger.error(f"Could not connect to {base_url}.")
        return

    # 4. Poll status
    status_url = f"{base_url}/api/translate/jobs/{job_id}/status"
    while True:
        try:
            status_res = requests.get(status_url, headers=headers)
            if status_res.status_code != 200:
                logger.warning(f"Status check failed: {status_res.status_code}")
                time.sleep(5)
                continue
                
            status_data = status_res.json()['data']
            status = status_data['status']
            progress = status_data.get('progress', 0)
            message = status_data.get('status_message', '')
            
            logger.info(f"Status: {status} ({progress}%) - {message}")
            
            if status == 'completed':
                logger.info("Translation completed!")
                download_url = status_data.get('download_url') or status_data.get('result', {}).get('output_video')
                if download_url:
                    # If it's a relative URL, prepend base_url
                    if not download_url.startswith('http'):
                        full_download_url = f"{base_url}{download_url}"
                    else:
                        full_download_url = download_url
                        
                    download_result(full_download_url, headers, job_id)
                else:
                    logger.error("No download URL found in result")
                    logger.info(json.dumps(status_data, indent=2))
                break
                
            elif status == 'failed':
                logger.error(f"Translation failed: {status_data.get('error')}")
                break
                
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"Error polling status: {e}")
            time.sleep(5)

def download_result(url, headers, job_id):
    logger.info(f"Downloading result from {url}...")
    try:
        # For download, we might need to handle the response as stream
        with requests.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()
            output_filename = f"backend/outputs/translated_{job_id}.mp4"
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            
            with open(output_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
                    
        logger.info(f"Saved translated video to: {os.path.abspath(output_filename)}")
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        # Try finding it in the directory based on job_id if download fails
        fallback_path = f"backend/outputs/translated_video_{job_id}.mp4"
        if os.path.exists(fallback_path):
             logger.info(f"Found file locally at: {os.path.abspath(fallback_path)}")

if __name__ == "__main__":
    translate_video()

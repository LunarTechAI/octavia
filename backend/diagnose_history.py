import requests
import json
import os

# Configuration
API_URL = "http://127.0.0.1:8000"
# We need a token to test authenticated endpoints. 
# Since I don't have a valid user token, I'll try to check the local files directly first.

def check_local_files():
    files = ["jobs_db.json", "translation_jobs.json", "subtitle_jobs.json"]
    print("--- Checking Local Sync Files ---")
    for f in files:
        path = os.path.join("c:\\Users\\onyan\\octavia\\octavia\\backend", f)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"[OK] {f} exists ({size} bytes)")
            try:
                with open(path, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                    print(f"     Items count: {len(data)}")
                    # Print first item ID for verification
                    if data:
                        first_id = list(data.keys())[0]
                        print(f"     First item ID: {first_id}")
                        print(f"     First item user_id: {data[first_id].get('user_id')}")
            except Exception as e:
                print(f"[ERR] Failed to parse {f}: {e}")
                # Try to read raw content to debug
                try:
                    with open(path, 'rb') as rb:
                        print(f"     Raw content (first 20 bytes): {rb.read(20)}")
                except:
                    pass
        else:
            print(f"[MISSING] {f}")

if __name__ == "__main__":
    check_local_files()

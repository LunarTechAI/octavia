"""
Migration script to move existing JSON jobs to Supabase
Run this after applying SQL migration
"""
import os
import sys
import json
import asyncio
from pathlib import Path

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[WARNING] python-dotenv not installed, environment variables may not be loaded")

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared_dependencies import supabase


def map_job_fields(job_data):
    """Map old field names to new database schema"""
    
    # Map target_lang -> target_language
    if 'target_lang' in job_data and 'target_language' not in job_data:
        job_data['target_language'] = job_data.pop('target_lang')
    
    # Map format -> output_format (for subtitles)
    if 'format' in job_data and 'output_format' not in job_data:
        # Only move format to output_format if it's a subtitle type
        if job_data.get('type') in ['subtitles', 'subtitle_to_audio']:
            job_data['output_format'] = job_data.pop('format')
    
    return job_data


def get_or_create_user(user_id, user_email):
    """Get user from database or create if doesn't exist"""
    if not user_email:
        return None
    
    try:
        # Try to find user by email in auth.users table (not users!)
        response = supabase.table("users").select("*").eq("email", user_email).execute()
        
        if response.data and len(response.data) > 0:
            existing_user = response.data[0]
            print(f"  [INFO] Found existing user: {existing_user['id']}")
            return existing_user['id']
        
        # User not found, skip this job
        print(f"  [SKIP] User not found for: {user_email}")
        return None
            
    except Exception as e:
        print(f"  [WARNING] Error handling user: {e}")
        return None


async def migrate_jobs_from_json():
    """Migrate jobs from JSON files to Supabase"""
    
    # Check if Supabase is initialized
    if supabase is None:
        print("[ERROR] Supabase client not initialized!")
        print("\nPlease set your environment variables:")
        print("  export SUPABASE_URL='your-supabase-url'")
        print("  export SUPABASE_SERVICE_KEY='your-service-role-key'")
        print("\nOr add them to your .env file")
        return
    
    total_migrated = 0
    total_failed = 0
    
    # Migration sources
    sources = [
        ("jobs_db.json", "video"),
        ("subtitle_jobs.json", "subtitle_to_audio"),
        ("translation_jobs.json", "translation")
    ]
    
    for filename, job_type in sources:
        filepath = Path(__file__).parent.parent / filename
        
        if not filepath.exists():
            print(f"[SKIP] {filename} not found")
            continue
        
        # Check if file is empty
        if filepath.stat().st_size == 0:
            print(f"[SKIP] {filename} is empty (0 bytes)")
            continue
        
        print(f"\n[MIGRATING] {filename}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                jobs_data = json.load(f)
            
            # Check if data is a dict
            if not isinstance(jobs_data, dict):
                print(f"[SKIP] {filename} does not contain valid job data")
                continue
            
            for job_id, job_data in jobs_data.items():
                try:
                    # Ensure job has id
                    if 'id' not in job_data:
                        job_data['id'] = job_id
                    
                    # Set job type if not present
                    if 'type' not in job_data:
                        job_data['type'] = job_type
                    
                    # Set default values
                    if 'version' not in job_data:
                        job_data['version'] = 0
                    if 'metrics' not in job_data:
                        job_data['metrics'] = {}
                    if 'result' not in job_data:
                        job_data['result'] = {}
                    
                    # Map field names to match new schema
                    job_data = map_job_fields(job_data)
                    
                    # Handle user foreign key
                    user_id = job_data.get('user_id')
                    user_email = job_data.get('user_email')
                    
                    if user_id and user_email:
                        # Ensure user exists
                        validated_user_id = get_or_create_user(user_id, user_email)
                        job_data['user_id'] = validated_user_id
                    
                    # Insert into Supabase
                    result = supabase.table("translation_jobs").upsert(job_data, on_conflict="id").execute()
                    
                    if result.data:
                        total_migrated += 1
                        print(f"  [OK] {job_id} - {job_data.get('status', 'unknown')}")
                    else:
                        total_failed += 1
                        print(f"  [FAIL] {job_id}")
                
                except Exception as e:
                    total_failed += 1
                    print(f"  [ERROR] {job_id}: {e}")
        
        except Exception as e:
            print(f"[ERROR] Failed to load {filename}: {e}")
    
    print(f"\n[MIGRATION COMPLETE]")
    print(f"Total migrated: {total_migrated}")
    print(f"Total failed: {total_failed}")
    
    # Create backup of JSON files
    print(f"\n[BACKUP] Creating backups of JSON files...")
    for filename, _ in sources:
        filepath = Path(__file__).parent.parent / filename
        if filepath.exists():
            backup_path = filepath.with_suffix('.json.backup')
            import shutil
            shutil.copy(filepath, backup_path)
            print(f"  [BACKUP] {filename} -> {backup_path.name}")
 

if __name__ == "__main__":
    print("=" * 60)
    print("Job Migration to Supabase")
    print("=" * 60)
    
    asyncio.run(migrate_jobs_from_json())

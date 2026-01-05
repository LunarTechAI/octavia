"""
Migration script to move existing JSON jobs to Supabase
Run this after applying SQL migration
"""
import os
import sys
import json
import asyncio
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[WARNING] python-dotenv not installed")

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared_dependencies import supabase


def map_job_fields(job_data):
    """Map old field names to new database schema"""
    
    # Map target_lang -> target_language
    if 'target_lang' in job_data and 'target_language' not in job_data:
        job_data['target_language'] = job_data.pop('target_lang')
    
    # Map format -> output_format (for subtitles)
    if 'format' in job_data and 'output_format' not in job_data:
        if job_data.get('type') in ['subtitles', 'subtitle_to_audio']:
            job_data['output_format'] = job_data.pop('format')
    
    return job_data


async def migrate_jobs_from_json():
    """Migrate jobs from JSON files to Supabase"""
    
    if supabase is None:
        print("[ERROR] Supabase client not initialized!")
        return
    
    total_migrated = 0
    total_failed = 0
    
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
        
        if filepath.stat().st_size == 0:
            print(f"[SKIP] {filename} is empty (0 bytes)")
            continue
        
        print(f"\n[MIGRATING] {filename}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                jobs_data = json.load(f)
            
            if not isinstance(jobs_data, dict):
                print(f"[SKIP] {filename} does not contain valid job data")
                continue
            
            for job_id, job_data in jobs_data.items():
                try:
                    if 'id' not in job_data:
                        job_data['id'] = job_id
                    
                    if 'type' not in job_data:
                        job_data['type'] = job_type
                    
                    if 'version' not in job_data:
                        job_data['version'] = 0
                    if 'metrics' not in job_data:
                        job_data['metrics'] = {}
                    if 'result' not in job_data:
                        job_data['result'] = {}
                    
                    job_data = map_job_fields(job_data)
                    
                    # Remove language field if present (not in DB schema)
                    job_data.pop('language', None)
                    
                    # Try insert/upsert
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

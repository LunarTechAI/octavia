"""
Temporary file cleanup utilities for Octavia.

This module provides automatic cleanup of temporary files created during
video, audio, and subtitle processing jobs.
"""
import os
import glob
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

# Temp file patterns - regex-style patterns for matching temp files
TEMP_FILE_PATTERNS = [
    r"temp_audio_*",           # Audio translation temp files
    r"temp_video_*",           # Video translation temp files  
    r"temp_subtitle_*",        # Subtitle processing temp files
    r"temp_subtitle_audio_*",  # Subtitle-to-audio temp WAV files
    r"preview_*",              # Voice preview files
    r"subtitles_*.srt",        # Generated subtitle files (older style)
    r"translated_subtitle_*.srt",  # Translated subtitle files
    r"translated_audio_*.wav", # Translated audio files (temp)
    r"translated_audio_*.mp3", # Translated audio files (temp)
    r"subtitle_audio_*.mp3",   # Subtitle-to-audio output files
    r"backend/temp_video_*",   # Video files in backend dir
    r"backend/temp_audio_*",   # Audio files in backend dir
    r"backend/temp_subtitle_*", # Subtitle files in backend dir
]

# Directories to scan for temp files
TEMP_DIRS = [
    "",                        # Root project directory
    "backend/",               # Backend directory
    "outputs/",               # Outputs directory
    "backend/outputs/",       # Backend outputs
]

# File age threshold for cleanup (older files are cleaned first)
MAX_FILE_AGE_HOURS = 24  # Clean files older than 24 hours by default


def is_temp_file(filename: str) -> bool:
    """
    Check if a filename matches any of the temp file patterns.
    
    Args:
        filename: The filename to check
        
    Returns:
        True if filename matches a temp pattern, False otherwise
    """
    name = os.path.basename(filename)
    
    for pattern in TEMP_FILE_PATTERNS:
        # Convert glob pattern to fnmatch pattern
        if pattern.endswith("*"):
            prefix = pattern[:-1]  # Remove the *
            if name.startswith(prefix):
                return True
        else:
            import fnmatch
            if fnmatch.fnmatch(name, pattern):
                return True
    
    return False


def get_temp_files_in_directory(directory: str = "") -> List[str]:
    """
    Get all temp files in a specified directory.
    
    Args:
        directory: Directory path to scan (empty string = current dir)
        
    Returns:
        List of paths to temp files
    """
    temp_files = []
    base_dir = directory.rstrip("/")
    
    if not os.path.exists(base_dir):
        return temp_files
    
    for pattern in TEMP_FILE_PATTERNS:
        # Handle patterns with directory prefix
        if "/" in pattern:
            search_pattern = os.path.join(base_dir, pattern) if base_dir else pattern
        else:
            search_pattern = os.path.join(base_dir, "*", pattern) if base_dir else pattern
        
        # Use glob to find matching files
        if "*" in pattern:
            if base_dir:
                full_pattern = os.path.join(base_dir, pattern)
            else:
                full_pattern = pattern
            
            matches = glob.glob(full_pattern)
            temp_files.extend(matches)
        else:
            # Direct file match
            file_path = os.path.join(base_dir, pattern) if base_dir else pattern
            if os.path.exists(file_path):
                temp_files.append(file_path)
    
    return list(set(temp_files))  # Remove duplicates


def cleanup_temp_files(
    directory: str = "",
    max_age_hours: int = MAX_FILE_AGE_HOURS,
    patterns: Optional[List[str]] = None,
    exclude_paths: Optional[Set[str]] = None
) -> dict:
    """
    Clean up temporary files from a directory.
    
    This function scans the specified directory and removes temporary files
    created by the Octavia video translation system.
    
    Args:
        directory: Directory to clean (empty = current directory)
        max_age_hours: Only clean files older than this many hours
        patterns: Optional list of patterns to use (defaults to TEMP_FILE_PATTERNS)
        exclude_paths: Set of file paths to exclude from cleanup
        
    Returns:
        Dictionary with cleanup statistics:
        {
            "scanned_dir": str,
            "total_found": int,
            "cleaned": int,
            "failed": int,
            "skipped": int,
            "freed_bytes": int,
            "details": List[str]
        }
    """
    exclude_paths = exclude_paths or set()
    patterns = patterns or TEMP_FILE_PATTERNS
    
    now = datetime.now()
    cutoff_time = now - timedelta(hours=max_age_hours)
    
    result = {
        "scanned_dir": directory or ".",
        "total_found": 0,
        "cleaned": 0,
        "failed": 0,
        "skipped": 0,
        "freed_bytes": 0,
        "details": []
    }
    
    for pattern in patterns:
        # Build glob pattern
        if "/" in pattern:
            if directory:
                glob_pattern = os.path.join(directory, pattern)
            else:
                glob_pattern = pattern
        else:
            if directory:
                glob_pattern = os.path.join(directory, pattern)
            else:
                glob_pattern = pattern
        
        # Find matching files
        try:
            matching_files = glob.glob(glob_pattern)
        except Exception as e:
            logger.warning(f"Error scanning pattern {pattern}: {e}")
            continue
        
        for file_path in matching_files:
            if file_path in exclude_paths:
                result["skipped"] += 1
                continue
            
            try:
                if not os.path.exists(file_path):
                    result["skipped"] += 1
                    continue
                
                # Check file age
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if file_mtime > cutoff_time:
                    # File is too new, skip
                    result["skipped"] += 1
                    continue
                
                # Get file size before deletion
                file_size = os.path.getsize(file_path)
                
                # Delete the file
                os.remove(file_path)
                
                result["cleaned"] += 1
                result["freed_bytes"] += file_size
                result["details"].append(f"Removed: {file_path} ({file_size} bytes)")
                
            except Exception as e:
                result["failed"] += 1
                result["details"].append(f"Failed to remove {file_path}: {e}")
                logger.error(f"Error cleaning temp file {file_path}: {e}")
    
    result["total_found"] = result["cleaned"] + result["skipped"] + result["failed"]
    
    return result


def cleanup_specific_files(file_paths: List[str]) -> dict:
    """
    Clean up a specific list of files.
    
    Args:
        file_paths: List of file paths to remove
        
    Returns:
        Dictionary with cleanup results
    """
    result = {
        "total": len(file_paths),
        "cleaned": 0,
        "failed": 0,
        "freed_bytes": 0,
        "details": []
    }
    
    for file_path in file_paths:
        try:
            if not os.path.exists(file_path):
                result["details"].append(f"Skipped (not exists): {file_path}")
                continue
            
            file_size = os.path.getsize(file_path)
            os.remove(file_path)
            
            result["cleaned"] += 1
            result["freed_bytes"] += file_size
            result["details"].append(f"Removed: {file_path}")
            
        except Exception as e:
            result["failed"] += 1
            result["details"].append(f"Failed: {file_path} - {e}")
            logger.error(f"Error removing file {file_path}: {e}")
    
    return result


def cleanup_old_jobs_directory(base_path: str = "backend/outputs") -> dict:
    """
    Clean up old job outputs that may have been orphaned.
    
    Args:
        base_path: Base path for job outputs
        
    Returns:
        Cleanup results dictionary
    """
    result = {
        "scanned": base_path,
        "cleaned": 0,
        "failed": 0,
        "freed_bytes": 0,
        "details": []
    }
    
    if not os.path.exists(base_path):
        return result
    
    now = datetime.now()
    cutoff_time = now - timedelta(hours=48)  # Older threshold for job outputs
    
    try:
        # Scan directory for old job folders
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            
            if not os.path.isdir(item_path):
                continue
            
            # Check if folder is old
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(item_path))
                if mtime < cutoff_time:
                    # Check if it looks like a job folder (UUID format)
                    if len(item) == 36:  # UUID length
                        # This is likely an old job folder
                        size = 0
                        for root, dirs, files in os.walk(item_path):
                            for f in files:
                                fpath = os.path.join(root, f)
                                try:
                                    size += os.path.getsize(fpath)
                                    os.remove(fpath)
                                except:
                                    pass
                        
                        # Remove directory
                        try:
                            os.rmdir(item_path)
                            result["cleaned"] += 1
                            result["freed_bytes"] += size
                            result["details"].append(f"Cleaned old job: {item}")
                        except Exception as e:
                            result["failed"] += 1
                            logger.error(f"Error removing old job folder {item_path}: {e}")
            except Exception as e:
                logger.warning(f"Error checking item {item}: {e}")
    
    except Exception as e:
        logger.error(f"Error scanning job outputs: {e}")
    
    return result


def cleanup_orphaned_files(working_job_ids: Set[str]) -> dict:
    """
    Clean up files for jobs that no longer exist.
    
    Args:
        working_job_ids: Set of currently active job IDs
        
    Returns:
        Cleanup results
    """
    result = {
        "cleaned": 0,
        "failed": 0,
        "details": []
    }
    
    # Pattern to match: files with UUID-like names
    import re
    
    for directory in ["backend/outputs", "outputs"]:
        if not os.path.exists(directory):
            continue
        
        for filename in os.listdir(directory):
            # Check if filename contains a job ID that we don't recognize
            for job_id in working_job_ids:
                if job_id in filename:
                    break
            else:
                # This file doesn't belong to any known job
                file_path = os.path.join(directory, filename)
                
                try:
                    # Only clean if file is old (older than 1 hour)
                    mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if datetime.now() - mtime > timedelta(hours=1):
                        os.remove(file_path)
                        result["cleaned"] += 1
                        result["details"].append(f"Cleaned orphaned: {file_path}")
                except Exception as e:
                    result["failed"] += 1
                    logger.warning(f"Error cleaning orphaned file {file_path}: {e}")
    
    return result


def run_full_cleanup() -> dict:
    """
    Run a comprehensive cleanup across all temp directories.
    
    This is typically called on server startup or shutdown.
    
    Returns:
        Combined cleanup results from all directories
    """
    all_results = {
        "total_cleaned": 0,
        "total_failed": 0,
        "total_freed_bytes": 0,
        "by_directory": {},
        "timestamp": datetime.now().isoformat()
    }
    
    for directory in TEMP_DIRS:
        if os.path.exists(directory) or directory == "":
            result = cleanup_temp_files(directory=directory)
            all_results["by_directory"][directory] = result
            all_results["total_cleaned"] += result["cleaned"]
            all_results["total_failed"] += result["failed"]
            all_results["total_freed_bytes"] += result["freed_bytes"]
    
    # Clean old job outputs
    job_result = cleanup_old_jobs_directory()
    all_results["by_directory"]["old_jobs"] = job_result
    all_results["total_cleaned"] += job_result["cleaned"]
    all_results["total_freed_bytes"] += job_result["freed_bytes"]
    
    logger.info(
        f"Full cleanup completed: {all_results['total_cleaned']} files cleaned, "
        f"{all_results['total_freed_bytes'] / 1024:.1f} KB freed"
    )
    
    return all_results


def format_bytes(num_bytes: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


if __name__ == "__main__":
    # Run cleanup when executed directly
    print("Running temp file cleanup...")
    print("-" * 50)
    
    result = run_full_cleanup()
    
    print(f"Total files cleaned: {result['total_cleaned']}")
    print(f"Total space freed: {format_bytes(result['total_freed_bytes'])}")
    print(f"Failed: {result['total_failed']}")
    
    if result['details']:
        print("\nDetails:")
        for detail in result['details'][:10]:  # Show first 10
            print(f"  - {detail}")

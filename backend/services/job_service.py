from typing import Dict, Any, Optional
import logging
from datetime import datetime
from services.job_storage import job_storage

logger = logging.getLogger(__name__)

class JobService:
    """Service for managing translation jobs"""

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self, job_id: str, job_data: Dict[str, Any]) -> None:
        """Create a new job"""
        self._jobs[job_id] = job_data
        logger.info(f"Created job {job_id}")

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job data"""
        return self._jobs.get(job_id)

    def update_job_status(self, job_id: str, status: str) -> None:
        """Update job status"""
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = status
            logger.debug(f"Updated job {job_id} status to {status}")

    def update_job_progress(self, job_id: str, progress: int, message: str = "") -> None:
        """Update job progress"""
        if job_id in self._jobs:
            self._jobs[job_id]["progress"] = progress
            if message:
                self._jobs[job_id]["message"] = message
            logger.debug(f"Updated job {job_id} progress to {progress}")

    def complete_job(self, job_id: str, result: Dict[str, Any]) -> None:
        """Mark job as completed with results"""
        if job_id in self._jobs:
            self._jobs[job_id].update({
                "status": "completed",
                "progress": 100,
                "result": result,
                "completed_at": datetime.utcnow().isoformat()
            })
            logger.info(f"Completed job {job_id}")

    def fail_job(self, job_id: str, error: str) -> None:
        """Mark job as failed"""
        if job_id in self._jobs:
            self._jobs[job_id].update({
                "status": "failed",
                "error": error,
                "progress": 0,
                "failed_at": datetime.utcnow().isoformat()
            })
            logger.error(f"Failed job {job_id}: {error}")

    def get_all_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Get all jobs (for debugging)"""
        return self._jobs.copy()

# Global instance
job_service = JobService()
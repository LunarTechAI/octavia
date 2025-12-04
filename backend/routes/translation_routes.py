"""
API routes for translation features
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import os
import uuid
import json
from datetime import datetime
from modules.subtitle_generator import SubtitleGenerator
from modules.audio_translator import AudioTranslator
from modules.pipeline import VideoTranslationPipeline

router = APIRouter(prefix="/api/translate", tags=["translation"])

# In-memory job tracking
translation_jobs = {}

@router.post("/subtitles")
async def generate_subtitles(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    format: str = Form("srt")
):
    """Generate subtitles from video/audio file"""
    try:
        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1]
        file_path = f"temp_{file_id}{file_ext}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Initialize subtitle generator
        generator = SubtitleGenerator()
        
        # Process file
        result = generator.process_file(file_path, format, language)
        
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return {
            "success": True,
            "job_id": file_id,
            "download_url": f"/api/download/subtitles/{file_id}",
            "format": format,
            "segment_count": result["segment_count"],
            "language": result["language"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Subtitle generation failed: {str(e)}")

@router.post("/audio")
async def translate_audio(
    file: UploadFile = File(...),
    source_lang: str = Form("en"),
    target_lang: str = Form("es")
):
    """Translate audio file to another language"""
    try:
        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1]
        file_path = f"temp_{file_id}{file_ext}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Initialize translator
        translator = AudioTranslator(source_lang, target_lang)
        
        # Process audio
        result = translator.process_audio(file_path)
        
        # Store job info
        translation_jobs[file_id] = {
            "id": file_id,
            "type": "audio",
            "status": "completed",
            "result": result,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return {
            "success": True,
            "job_id": file_id,
            "download_url": f"/api/download/audio/{file_id}",
            "duration_match_percent": result["duration_match_percent"],
            "speed_adjustment": result["speed_adjustment"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio translation failed: {str(e)}")

@router.post("/video/enhanced")
async def translate_video_enhanced(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_language: str = Form("es"),
    chunk_size: int = Form(30)
):
    """Enhanced video translation with chunk processing"""
    try:
        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1]
        file_path = f"temp_{file_id}{file_ext}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Create job entry
        job_id = str(uuid.uuid4())
        translation_jobs[job_id] = {
            "id": job_id,
            "type": "video_enhanced",
            "status": "processing",
            "progress": 0,
            "file_path": file_path,
            "target_language": target_language,
            "chunk_size": chunk_size,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Process in background
        background_tasks.add_task(
            process_video_enhanced_job,
            job_id,
            file_path,
            target_language,
            chunk_size
        )
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "Video translation started in background",
            "status_url": f"/api/jobs/{job_id}/status"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video translation failed: {str(e)}")

async def process_video_enhanced_job(job_id, file_path, target_language, chunk_size):
    """Background task for enhanced video translation"""
    try:
        # Update job status
        translation_jobs[job_id]["progress"] = 10
        
        # Initialize pipeline
        pipeline = VideoTranslationPipeline()
        pipeline.load_models()
        
        # Update progress
        translation_jobs[job_id]["progress"] = 30
        
        # Process video
        result = pipeline.process_video(file_path, target_language)
        
        # Update job with results
        translation_jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "result": result,
            "completed_at": datetime.utcnow().isoformat(),
            "output_path": result["output_path"]
        })
        
        # Cleanup temp file
        if os.path.exists(file_path):
            os.remove(file_path)
            
    except Exception as e:
        translation_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.utcnow().isoformat()
        })

@router.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Get status of a translation job"""
    if job_id not in translation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = translation_jobs[job_id]
    
    response = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0),
        "type": job["type"],
        "created_at": job["created_at"]
    }
    
    if job["status"] == "completed":
        response.update({
            "completed_at": job.get("completed_at"),
            "result": job.get("result", {})
        })
    elif job["status"] == "failed":
        response.update({
            "error": job.get("error"),
            "failed_at": job.get("failed_at")
        })
    
    return response

@router.get("/download/{file_type}/{file_id}")
async def download_file(file_type: str, file_id: str):
    """Download generated files"""
    if file_type == "subtitles":
        filename = "subtitles.srt"
        media_type = "text/plain"
    elif file_type == "audio":
        filename = "translated_audio.wav"
        media_type = "audio/wav"
    elif file_type == "video":
        filename = "translated_video.mp4"
        media_type = "video/mp4"
    else:
        raise HTTPException(status_code=404, detail="File type not found")
    
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        filename,
        media_type=media_type,
        filename=f"octavia_{file_type}_{file_id}{os.path.splitext(filename)[1]}"
    )
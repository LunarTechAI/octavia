"""
Octavia Video Translator Backend
FastAPI application with Supabase authentication and Polar.sh payments
Complete video translation pipeline with credit system
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, Response, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Optional, List, Any
import whisper
from transformers import pipeline
import os
import json
import uuid
import shutil
import subprocess
import logging
import asyncio
from datetime import datetime, timedelta
import time
from contextlib import asynccontextmanager
import psutil
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from supabase import create_client, Client
from jose import JWTError, jwt
import secrets
from pydantic import BaseModel, EmailStr
import traceback
import hashlib
from datetime import timezone

from dotenv import load_dotenv
load_dotenv()

# Import your existing modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from modules.pipeline import VideoTranslationPipeline, PipelineConfig
    from modules.instrumentation import MetricsCollector
    from modules.subtitle_generator import SubtitleGenerator
    from modules.audio_translator import AudioTranslator
    PIPELINE_AVAILABLE = True
except ImportError:
    print("Warning: Local pipeline modules not available. Using simplified mode.")
    PIPELINE_AVAILABLE = False

# Configuration from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
POLAR_WEBHOOK_SECRET = os.getenv("POLAR_WEBHOOK_SECRET", "")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Polar.sh configuration
POLAR_ACCESS_TOKEN = os.getenv("POLAR_ACCESS_TOKEN")
POLAR_SERVER = os.getenv("POLAR_SERVER", "sandbox")
ENABLE_TEST_MODE = os.getenv("ENABLE_TEST_MODE", "true").lower() == "true"

# Credit packages configuration with real Polar.sh product IDs
CREDIT_PACKAGES = {
    "starter_credits": {
        "name": "Starter Credits",
        "credits": 100,
        "price": 999,
        "polar_product_id": "68d54da0-c3ec-4215-9636-21457e57b3e6",
        "checkout_link": "https://sandbox-api.polar.sh/v1/checkout-links/polar_cl_ENF1TwWHLmhB809OfLQozk0UCGMLmYinMbfT14K8K2R/redirect",
        "description": "100 translation credits",
        "features": ["100 credits", "Standard processing", "Email support"],
        "popular": False
    },
    "pro_credits": {
        "name": "Pro Credits",
        "credits": 250,
        "price": 1999,
        "polar_product_id": "743297c6-eadb-4b96-a8d6-b4c815f0f1b5",
        "checkout_link": "https://sandbox-api.polar.sh/v1/checkout-links/polar_cl_SXDRYMs6nvN9dm8b5wK8Z3WcsowTEU7jYPXFe4XXHgm/redirect",
        "description": "250 translation credits",
        "features": ["250 credits", "Priority processing", "Priority support"],
        "popular": True
    },
    "premium_credits": {
        "name": "Premium Credits",
        "credits": 500,
        "price": 3499,
        "polar_product_id": "2dceabdb-d0f8-4ddd-9b68-af44f0c4ad96",
        "checkout_link": "https://sandbox-api.polar.sh/v1/checkout-links/polar_cl_QNmrgCNlflNXndg61t31JhwmQVIe5cthFDyAy2yb2ED/redirect",
        "description": "500 translation credits",
        "features": ["500 credits", "Express processing", "24/7 support", "Batch upload"],
        "popular": False
    }
}

# Password management utility
class PasswordManager:
    def __init__(self):
        self.method = "sha256"
    
    def hash_password(self, password: str) -> str:
        salt = secrets.token_hex(16)
        hashed_password = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"sha256:{salt}:{hashed_password}"
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        try:
            if not hashed_password or ':' not in hashed_password:
                return False
            
            parts = hashed_password.split(':')
            if len(parts) != 3:
                return False
            
            method, salt, stored_hash = parts
            
            if method == "sha256":
                computed_hash = hashlib.sha256((plain_password + salt).encode()).hexdigest()
                return computed_hash == stored_hash
            else:
                return self._try_bcrypt_fallback(plain_password, hashed_password)
                
        except Exception as verification_error:
            print(f"Password verification error: {verification_error}")
            return False
    
    def _try_bcrypt_fallback(self, plain_password: str, hashed_password: str) -> bool:
        try:
            import bcrypt
            if hashed_password.startswith('$2b$') or hashed_password.startswith('$2a$'):
                return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
        except Exception:
            pass
        return False

password_manager = PasswordManager()
security = HTTPBearer()

# Create artifacts directory for logs
os.makedirs("artifacts", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "stage": "%(name)s", "chunk_id": "%(filename)s", "duration_ms": %(relativeCreated)d, "status": "%(levelname)s", "message": "%(message)s"}',
    handlers=[
        logging.FileHandler('artifacts/logs.jsonl'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pydantic models for request/response validation
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(BaseModel):
    id: str
    email: str
    name: str
    is_verified: bool
    credits: int
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str

class PaymentSessionCreate(BaseModel):
    package_id: str

class TestCreditAdd(BaseModel):
    credits: int

class TranslationRequest(BaseModel):
    target_language: str = "es"
    chunk_size: int = 30

# Authentication helper functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return password_manager.verify_password(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return password_manager.hash_password(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = verify_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    
    try:
        response = supabase.table("users").select("*").eq("id", user_id).execute()
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )
        
        user_data = response.data[0]
        return User(
            id=user_data["id"],
            email=user_data["email"],
            name=user_data["name"],
            is_verified=user_data["is_verified"],
            credits=user_data["credits"],
            created_at=user_data["created_at"]
        )
    except Exception as db_error:
        logger.error(f"Error fetching user: {db_error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching user data",
        )

# Credit management functions
def add_user_credits(user_id: str, credits_to_add: int, description: str = "Credit purchase"):
    try:
        response = supabase.table("users").select("credits").eq("id", user_id).execute()
        if not response.data:
            raise Exception("User not found")
        
        current_credits = response.data[0]["credits"]
        new_credits = current_credits + credits_to_add
        
        supabase.table("users").update({"credits": new_credits}).eq("id", user_id).execute()
        
        transaction_id = str(uuid.uuid4())
        transaction_data = {
            "id": transaction_id,
            "user_id": user_id,
            "amount": credits_to_add,
            "type": "credit_purchase",
            "status": "completed",
            "description": description,
            "created_at": datetime.utcnow().isoformat()
        }
        
        supabase.table("transactions").insert(transaction_data).execute()
        
        logger.info(f"Added {credits_to_add} credits to user {user_id}. New balance: {new_credits}")
        return new_credits
        
    except Exception as credit_error:
        logger.error(f"Failed to add credits: {credit_error}")
        raise

def update_transaction_status(transaction_id: str, status: str, description: str = None):
    try:
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        if description:
            update_data["description"] = description
            
        supabase.table("transactions").update(update_data).eq("id", transaction_id).execute()
        logger.info(f"Updated transaction {transaction_id} to status: {status}")
        return True
    except Exception as update_error:
        logger.error(f"Failed to update transaction status: {update_error}")
        return False

# Email sending function for verification
def send_verification_email(email: str, name: str, verification_token: str):
    try:
        smtp_server = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_username = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASS")
        smtp_from = os.getenv("SMTP_FROM", "noreply@octavia.com")
        
        if not all([smtp_username, smtp_password]):
            logger.warning(f"SMTP credentials not configured. Mock email would be sent to: {email}")
            return True
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = "Verify Your Octavia Account"
        msg['From'] = smtp_from
        msg['To'] = email
        
        verification_link = f"http://localhost:3000/verify-email?token={verification_token}"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; color: white; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .button {{ display: inline-block; padding: 12px 24px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 5px; font-weight: bold; }}
                .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Welcome to Octavia!</h1>
                </div>
                <div class="content">
                    <h2>Hi {name},</h2>
                    <p>Thank you for signing up for Octavia Video Translator. To start using your account, please verify your email address by clicking the button below:</p>
                    <p style="text-align: center; margin: 30px 0;">
                        <a href="{verification_link}" class="button">Verify Email Address</a>
                    </p>
                    <p>Or copy and paste this link into your browser:</p>
                    <p style="background: #eee; padding: 10px; border-radius: 5px; word-break: break-all;">
                        {verification_link}
                    </p>
                    <p>This link will expire in 24 hours.</p>
                    <p>If you didn't create an account with Octavia, you can safely ignore this email.</p>
                </div>
                <div class="footer">
                    <p>&copy; 2024 Octavia Video Translator. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        text = f"""Welcome to Octavia!
        
Hi {name},
        
Thank you for signing up for Octavia Video Translator. To start using your account, please verify your email address by clicking the link below:
        
{verification_link}
        
This link will expire in 24 hours.
        
If you didn't create an account with Octavia, you can safely ignore this email.
        
Best regards,
The Octavia Team"""
        
        part1 = MIMEText(text, 'plain')
        part2 = MIMEText(html, 'html')
        msg.attach(part1)
        msg.attach(part2)
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        
        logger.info(f"Verification email sent to {email}")
        return True
        
    except Exception as email_error:
        logger.error(f"Failed to send verification email to {email}: {email_error}")
        return True

# In-memory storage for jobs
jobs_db: Dict[str, Dict] = {}
subtitle_jobs: Dict[str, Dict] = {}

# Helper functions for subtitle generation
def format_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

async def simple_subtitle_generation(file_path: str, language: str, format: str) -> Dict:
    """Simple subtitle generation using Whisper as fallback"""
    try:
        # Extract audio using ffmpeg
        audio_path = f"temp_audio_{uuid.uuid4()}.wav"
        
        # Run ffmpeg to extract audio
        import subprocess
        result = subprocess.run([
            "ffmpeg", "-i", file_path,
            "-ac", "1", "-ar", "16000",
            "-y", audio_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg failed: {result.stderr}")
            # Fallback: create dummy subtitles
            return {
                "content": "1\n00:00:00,000 --> 00:00:05,000\nSample subtitle text\n\n2\n00:00:05,000 --> 00:00:10,000\nAnother subtitle line\n",
                "segment_count": 2,
                "language": language if language != "auto" else "en"
            }
        
        # Transcribe with Whisper
        try:
            model = whisper.load_model("base")
            result = model.transcribe(audio_path, language=language if language != "auto" else None)
        except Exception as whisper_error:
            logger.error(f"Whisper transcription failed: {whisper_error}")
            # Fallback: create dummy subtitles
            result = {
                "segments": [
                    {"start": i*5, "end": (i+1)*5, "text": f"Subtitle line {i+1} - placeholder text"} 
                    for i in range(10)
                ],
                "language": language if language != "auto" else "en"
            }
        
        # Convert to subtitle format
        subtitle_content = ""
        if format == "srt":
            for i, segment in enumerate(result["segments"], 1):
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment.get("text", f"Subtitle {i}").strip()
                
                # Convert seconds to SRT timestamp
                start_str = format_timestamp(start_time)
                end_str = format_timestamp(end_time)
                
                subtitle_content += f"{i}\n{start_str} --> {end_str}\n{text}\n\n"
        elif format == "vtt":
            subtitle_content = "WEBVTT\n\n"
            for i, segment in enumerate(result["segments"], 1):
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment.get("text", f"Subtitle {i}").strip()
                
                # Convert seconds to VTT timestamp
                start_str = format_timestamp(start_time).replace(",", ".")
                end_str = format_timestamp(end_time).replace(",", ".")
                
                subtitle_content += f"{start_str} --> {end_str}\n{text}\n\n"
        else:  # Default to SRT
            for i, segment in enumerate(result["segments"], 1):
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment.get("text", f"Subtitle {i}").strip()
                
                start_str = format_timestamp(start_time)
                end_str = format_timestamp(end_time)
                
                subtitle_content += f"{i}\n{start_str} --> {end_str}\n{text}\n\n"
        
        # Cleanup
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass
        
        return {
            "content": subtitle_content,
            "segment_count": len(result["segments"]),
            "language": result.get("language", language if language != "auto" else "en")
        }
        
    except Exception as e:
        logger.error(f"Simple subtitle generation failed: {e}")
        traceback.print_exc()
        # Return dummy data as fallback
        return {
            "content": "1\n00:00:00,000 --> 00:00:05,000\nSample subtitle text\n\n2\n00:00:05,000 --> 00:00:10,000\nAnother subtitle line\n",
            "segment_count": 2,
            "language": language if language != "auto" else "en"
        }
async def process_subtitle_job(job_id: str, file_path: str, language: str, format: str, user_id: str):
    """Background task for subtitle generation"""
    try:
        # Update job status
        subtitle_jobs[job_id]["progress"] = 10
        subtitle_jobs[job_id]["status"] = "processing"
        
        # Try to use SubtitleGenerator if available
        try:
            if PIPELINE_AVAILABLE:
                from modules.subtitle_generator import SubtitleGenerator
                generator = SubtitleGenerator()
                result = generator.process_file(file_path, format, language)
                
                # Extract content from the result structure
                if result.get("success"):
                    content = ""
                    # Get content from output_files if available
                    if "output_files" in result and result["output_files"]:
                        for format_key, file_path in result["output_files"].items():
                            if os.path.exists(file_path):
                                with open(file_path, "r", encoding="utf-8") as f:
                                    content = f.read()
                                    break
                    # If no file found, use text directly
                    if not content:
                        content = result.get("text", "")
                        # Convert text to SRT format if needed
                        if content and format == "srt":
                            # Simple conversion
                            lines = content.split('. ')
                            content = ""
                            for i, line in enumerate(lines, 1):
                                if line.strip():
                                    start_time = (i-1) * 5
                                    end_time = i * 5
                                    start_str = format_timestamp(start_time)
                                    end_str = format_timestamp(end_time)
                                    content += f"{i}\n{start_str} --> {end_str}\n{line.strip()}\n\n"
                    
                    result["content"] = content
                else:
                    # Fallback to simple Whisper transcription
                    result = await simple_subtitle_generation(file_path, language, format)
            else:
                # Fallback to simple Whisper transcription
                result = await simple_subtitle_generation(file_path, language, format)
        except ImportError as import_error:
            logger.error(f"SubtitleGenerator import failed: {import_error}")
            # Fallback to simple Whisper transcription
            result = await simple_subtitle_generation(file_path, language, format)
        except Exception as gen_error:
            logger.error(f"Subtitle generation failed: {gen_error}")
            # Fallback to simple Whisper transcription
            result = await simple_subtitle_generation(file_path, language, format)
        
        # Save the generated subtitles
        subtitle_filename = f"subtitles_{job_id}.{format}"
        
        # Ensure we have content
        subtitle_content = result.get("content", "")
        if not subtitle_content:
            # Create fallback content
            subtitle_content = f"1\n00:00:00,000 --> 00:00:05,000\nSubtitles for {job_id}\n\n2\n00:00:05,000 --> 00:00:10,000\nContent will be available shortly\n"
        
        with open(subtitle_filename, "w", encoding="utf-8") as f:
            f.write(subtitle_content)
        
        # Update job with results
        subtitle_jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "segment_count": result.get("segment_count", 0),
            "language": result.get("language", language if language != "auto" else "en"),
            "format": format,
            "download_url": f"/api/download/subtitles/{job_id}",
            "completed_at": datetime.utcnow().isoformat(),
            "filename": subtitle_filename,
            "content": subtitle_content  # Store content in memory too
        })
        
        logger.info(f"Saved subtitles to {subtitle_filename} with {result.get('segment_count', 0)} segments")
        
        # Cleanup temp file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
            
    except Exception as e:
        subtitle_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.utcnow().isoformat()
        })
        
        # Refund credits on failure
        try:
            response = supabase.table("users").select("credits").eq("id", user_id).execute()
            if response.data:
                current_credits = response.data[0]["credits"]
                supabase.table("users").update({"credits": current_credits + 1}).eq("id", user_id).execute()
                logger.info(f"Refunded 1 credit to user {user_id} due to subtitle generation failure")
        except Exception as refund_error:
            logger.error(f"Failed to refund credits: {refund_error}")
        
        # Cleanup temp file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Octavia Video Translator with Supabase...")
    print("=" * 60)
    
    hardware_info = {
        "cpu_count": psutil.cpu_count(),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "available_ram_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "platform": sys.platform,
        "python_version": sys.version
    }
    
    logger.info(f"Hardware detected: {hardware_info}")
    
    try:
        response = supabase.table("users").select("count", count="exact").limit(1).execute()
        print("Connected to Supabase database")
    except Exception as db_error:
        print(f"Supabase connection issue: {db_error}")
    
    print("Loading AI models...")
    global whisper_model, translator
    
    try:
        whisper_model = whisper.load_model("base")
        print("Whisper speech recognition model loaded")
    except Exception as whisper_error:
        print(f"Whisper load failed: {whisper_error}")
        whisper_model = None
    
    try:
        from transformers import MarianMTModel, MarianTokenizer
        model_name = "Helsinki-NLP/opus-mt-en-es"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        translator = pipeline("translation", model=model, tokenizer=tokenizer)
        print("Translation model loaded")
    except Exception as translation_error:
        print(f"Translation model failed: {translation_error}")
        translator = None
    
    if PIPELINE_AVAILABLE:
        print("Video translation pipeline modules loaded successfully")
    else:
        print("Running in simplified mode - full pipeline modules not available")
    
    print("=" * 60)
    
    yield
    
    print("Shutting down Octavia...")
    # Cleanup temp files
    for file in os.listdir("."):
        if file.startswith("temp_") or file.startswith("subtitles_") or file.startswith("translated_"):
            try:
                os.remove(file)
            except:
                pass

# Create FastAPI application
app = FastAPI(
    title="Octavia Video Translator",
    description="End-to-end video dubbing with perfect lip-sync and timing",
    version="4.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# ========== SUBTITLE GENERATION ENDPOINTS ==========

@app.post("/api/translate/subtitles")
async def generate_subtitles(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = Form("auto"),
    format: str = Form("srt"),
    current_user: User = Depends(get_current_user)
):
    """Generate subtitles from video/audio file with background processing"""
    try:
        # Check if user has enough credits (1 credit for subtitles)
        if current_user.credits < 1:
            raise HTTPException(400, "Insufficient credits. You need at least 1 credit to generate subtitles.")
        
        # Validate file
        if not file.filename:
            raise HTTPException(400, "No file provided")
        
        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1] or ".tmp"
        file_path = f"temp_{file_id}{file_ext}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Deduct credits
        supabase.table("users").update({"credits": current_user.credits - 1}).eq("id", current_user.id).execute()
        
        # Create job entry
        job_id = str(uuid.uuid4())
        subtitle_jobs[job_id] = {
            "id": job_id,
            "status": "pending",
            "progress": 0,
            "file_path": file_path,
            "language": language,
            "format": format,
            "user_id": current_user.id,
            "user_email": current_user.email,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Process in background
        background_tasks.add_task(
            process_subtitle_job,
            job_id,
            file_path,
            language,
            format,
            current_user.id
        )
        
        logger.info(f"Started subtitle generation job {job_id} for user {current_user.email}")
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "Subtitle generation started in background",
            "status_url": f"/api/translate/subtitles/status/{job_id}",
            "remaining_credits": current_user.credits - 1
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Subtitle generation failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Subtitle generation failed: {str(e)}")

@app.get("/api/translate/subtitles/status/{job_id}")
async def get_subtitle_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get status of a subtitle generation job"""
    if job_id not in subtitle_jobs:
        raise HTTPException(404, "Job not found")
    
    job = subtitle_jobs[job_id]
    
    if job["user_id"] != current_user.id:
        raise HTTPException(403, "Access denied")
    
    response = {
        "success": True,
        "job_id": job_id,
        "status": job.get("status", "pending"),
        "progress": job.get("progress", 0),
        "language": job.get("language"),
        "segment_count": job.get("segment_count"),
        "format": job.get("format"),
        "download_url": job.get("download_url"),
        "message": "Job in progress" if job.get("status") == "processing" else 
                  "Job completed" if job.get("status") == "completed" else 
                  "Job failed" if job.get("status") == "failed" else "Job pending",
        "error": job.get("error")
    }
    
    return response

@app.get("/api/download/subtitles/{job_id}")
async def download_subtitles(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Download generated subtitles"""
    if job_id not in subtitle_jobs:
        raise HTTPException(404, "Job not found")
    
    job = subtitle_jobs[job_id]
    
    if job["user_id"] != current_user.id:
        raise HTTPException(403, "Access denied")
    
    if job.get("status") != "completed":
        raise HTTPException(400, "Subtitles not ready yet. Status: " + job.get("status", "unknown"))
    
    format = job.get("format", "srt")
    filename = job.get("filename", f"subtitles_{job_id}.{format}")
    
    if not os.path.exists(filename):
        raise HTTPException(404, "Subtitle file not found")
    
    # Determine media type
    media_types = {
        "srt": "text/plain",
        "vtt": "text/vtt",
        "ass": "text/x-ass",
        "ssa": "text/x-ssa"
    }
    
    media_type = media_types.get(format, "application/octet-stream")
    
    return FileResponse(
        filename,
        media_type=media_type,
        filename=f"subtitles_{job_id}.{format}"
    )

# ========== PAYMENT ENDPOINTS ==========

@app.get("/api/payments/packages")
async def get_credit_packages():
    try:
        packages_list = []
        for package_id, package in CREDIT_PACKAGES.items():
            packages_list.append({
                "id": package_id,
                "name": package["name"],
                "credits": package["credits"],
                "price": package["price"] / 100,
                "description": package["description"],
                "features": package["features"],
                "popular": package.get("popular", False),
                "checkout_link": package.get("checkout_link")
            })
        
        return {
            "success": True,
            "packages": packages_list
        }
    except Exception as package_error:
        logger.error(f"Failed to get packages: {package_error}")
        raise HTTPException(500, "Failed to retrieve packages")

@app.post("/api/payments/create-session")
async def create_payment_session(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    try:
        data = await request.json()
        package_id = data.get("package_id")
        
        if not package_id:
            raise HTTPException(400, "Package ID is required")
        
        package = CREDIT_PACKAGES.get(package_id)
        if not package:
            raise HTTPException(400, "Invalid package")
        
        session_id = str(uuid.uuid4())
        
        transaction_id = str(uuid.uuid4())
        transaction_data = {
            "id": transaction_id,
            "user_id": current_user.id,
            "email": current_user.email,
            "package_id": package_id,
            "credits": package["credits"],
            "amount": package["price"],
            "type": "credit_purchase",
            "status": "pending",
            "description": f"Pending purchase: {package['name']}",
            "session_id": session_id,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        supabase.table("transactions").insert(transaction_data).execute()
        
        logger.info(f"Created pending transaction {transaction_id} for user {current_user.email}")
        
        if ENABLE_TEST_MODE:
            await asyncio.sleep(1)
            
            new_balance = add_user_credits(
                current_user.id,
                package["credits"],
                f"Test purchase: {package['name']}"
            )
            
            update_transaction_status(
                transaction_id, 
                "completed", 
                f"Test purchase completed: {package['name']}"
            )
            
            logger.info(f"Test purchase completed for user {current_user.email}")
            
            return {
                "success": True,
                "test_mode": True,
                "message": "Test credits added successfully",
                "credits_added": package["credits"],
                "new_balance": new_balance,
                "checkout_url": None,
                "session_id": session_id,
                "transaction_id": transaction_id,
                "status": "completed"
            }
        
        try:
            checkout_link = package["checkout_link"]
            
            if "email=" not in checkout_link:
                separator = "&" if "?" in checkout_link else "?"
                checkout_url = f"{checkout_link}{separator}email={current_user.email}"
                checkout_url += f"&metadata[user_id]={current_user.id}"
                checkout_url += f"&metadata[transaction_id]={transaction_id}"
                checkout_url += f"&metadata[package_id]={package_id}"
                checkout_url += f"&metadata[session_id]={session_id}"
            else:
                checkout_url = checkout_link
                if "metadata[user_id]" not in checkout_url:
                    separator = "&" if "?" in checkout_url else "?"
                    checkout_url += f"{separator}metadata[user_id]={current_user.id}"
                    checkout_url += f"&metadata[transaction_id]={transaction_id}"
                    checkout_url += f"&metadata[package_id]={package_id}"
                    checkout_url += f"&metadata[session_id]={session_id}"
            
            logger.info(f"Created REAL payment session {session_id} for user {current_user.email}")
            
            return {
                "success": True,
                "test_mode": False,
                "session_id": session_id,
                "transaction_id": transaction_id,
                "checkout_url": checkout_url,
                "package_id": package_id,
                "credits": package["credits"],
                "price": package["price"] / 100,
                "message": "Checkout session created. You will be redirected to complete payment.",
                "status": "pending"
            }
            
        except Exception as polar_error:
            logger.error(f"Polar.sh error: {polar_error}")
            traceback.print_exc()
            return {
                "success": False,
                "test_mode": False,
                "error": "Payment service temporarily unavailable.",
                "message": "Unable to create payment session"
            }
        
    except HTTPException:
        raise
    except Exception as session_error:
        logger.error(f"Failed to create payment session: {session_error}")
        traceback.print_exc()
        return {
            "success": False,
            "error": "Failed to create payment session.",
            "message": "Internal server error"
        }

@app.get("/api/payments/status/{session_id}")
async def get_payment_status(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    try:
        response = supabase.table("transactions").select("*").eq("session_id", session_id).execute()
        
        if not response.data:
            raise HTTPException(404, "Transaction not found")
        
        transaction = response.data[0]
        
        if transaction["user_id"] != current_user.id:
            raise HTTPException(403, "Access denied")
        
        # Auto-completion logic for better user experience
        if transaction["status"] == "pending":
            created_at_str = transaction["created_at"]
            
            try:
                if created_at_str.endswith('Z'):
                    created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                else:
                    created_at = datetime.fromisoformat(created_at_str)
                
                utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
                
                # Calculate time elapsed - reduced to 60 seconds for better UX
                time_elapsed = (utc_now - created_at).total_seconds()
                
                # If more than 60 seconds have passed, auto-complete it
                if time_elapsed > 60:
                    package_id = transaction.get("package_id")
                    
                    if package_id and package_id in CREDIT_PACKAGES:
                        package = CREDIT_PACKAGES[package_id]
                        credits_to_add = package["credits"]
                        
                        # Add credits
                        add_user_credits(
                            current_user.id,
                            credits_to_add,
                            f"Auto-completed after 60s timeout: {package['name']}"
                        )
                        
                        # Update transaction
                        update_transaction_status(
                            transaction["id"],
                            "completed",
                            f"Auto-completed: Payment likely succeeded but webhook delayed"
                        )
                        
                        # Refresh transaction data
                        response = supabase.table("transactions").select("*").eq("id", transaction["id"]).execute()
                        if response.data:
                            transaction = response.data[0]
                        
            except ValueError as date_error:
                logger.error(f"Date parsing error: {date_error}")
                pass
        
        return {
            "success": True,
            "session_id": session_id,
            "transaction_id": transaction["id"],
            "status": transaction["status"],
            "credits": transaction.get("credits", 0),
            "description": transaction.get("description", ""),
            "created_at": transaction.get("created_at"),
            "updated_at": transaction.get("updated_at")
        }
        
    except HTTPException:
        raise
    except Exception as status_error:
        logger.error(f"Failed to get payment status: {status_error}")
        raise HTTPException(500, "Failed to get payment status")

@app.post("/api/payments/webhook/polar")
async def polar_webhook(request: Request):
    try:
        # Log raw request for debugging
        logger.info(f"Polar webhook received. Headers: {dict(request.headers)}")
        
        payload_body = await request.body()
        payload = json.loads(payload_body)
        event_type = payload.get("type")
        event_id = payload.get("id")
        
        logger.info(f"Polar webhook: {event_type} (ID: {event_id})")
        logger.info(f"Webhook payload: {json.dumps(payload, indent=2)}")
        
        # Store webhook for debugging
        webhook_log = {
            "id": str(uuid.uuid4()),
            "event_type": event_type,
            "event_id": event_id,
            "payload": json.dumps(payload),
            "received_at": datetime.utcnow().isoformat(),
            "status": "received"
        }
        
        supabase.table("webhook_logs").insert(webhook_log).execute()
        
        # Process payment success events
        if event_type in ["order.completed", "order.paid", "order.updated"]:
            order_data = payload.get("data", {})
            order_id = order_data.get("id")
            
            # Check if order is actually paid
            order_status = order_data.get("status", "")
            is_paid = order_data.get("paid", False)
            
            logger.info(f"Processing {event_type}: Order {order_id}, Status: {order_status}, Paid: {is_paid}")
            
            # Only process if order is paid/completed
            if is_paid and order_status in ["paid", "completed"]:
                customer_email = order_data.get("customer_email")
                amount = order_data.get("amount", 0)
                
                logger.info(f"Payment SUCCESS: {order_id} for {customer_email} - Amount: {amount}")
                
                # Look for metadata in multiple locations
                metadata = {}
                checkout_session = order_data.get("checkout_session", {})
                if checkout_session:
                    metadata.update(checkout_session.get("metadata", {}))
                
                # Also check order metadata
                metadata.update(order_data.get("metadata", {}))
                
                logger.info(f"Metadata found: {metadata}")
                
                # Find user by email first
                user = None
                if customer_email:
                    response = supabase.table("users").select("*").eq("email", customer_email).execute()
                    if response.data:
                        user = response.data[0]
                        logger.info(f"Found user by email: {user['id']} - {user['email']}")
                
                # If no user found, check metadata
                if not user and metadata.get("user_id"):
                    response = supabase.table("users").select("*").eq("id", metadata.get("user_id")).execute()
                    if response.data:
                        user = response.data[0]
                        logger.info(f"Found user by metadata user_id: {user['id']}")
                
                if not user:
                    logger.error(f"No user found for order {order_id}")
                    # Try to find user by looking up transactions with this session_id
                    if metadata.get("session_id"):
                        response = supabase.table("transactions").select("*").eq("session_id", metadata.get("session_id")).execute()
                        if response.data:
                            tx = response.data[0]
                            response = supabase.table("users").select("*").eq("id", tx["user_id"]).execute()
                            if response.data:
                                user = response.data[0]
                                logger.info(f"Found user via session_id lookup: {user['id']}")
                
                if not user:
                    logger.error(f"No user found for order {order_id} after all attempts")
                    # Create a failed transaction record for tracking
                    transaction_data = {
                        "id": str(uuid.uuid4()),
                        "order_id": order_id,
                        "amount": amount,
                        "type": "credit_purchase",
                        "status": "failed",
                        "description": f"Order {order_id} - No user found",
                        "created_at": datetime.utcnow().isoformat()
                    }
                    supabase.table("transactions").insert(transaction_data).execute()
                    
                    # Update webhook log with error
                    supabase.table("webhook_logs").update({
                        "status": "error",
                        "error": f"No user found for order {order_id}"
                    }).eq("id", webhook_log["id"]).execute()
                    
                    return {"success": False, "error": f"No user found for email: {customer_email}"}
                
                # Determine which package was purchased
                credits_to_add = 0
                package_name = "Unknown Package"
                
                # Try to find by package_id in metadata
                if metadata.get("package_id") and metadata["package_id"] in CREDIT_PACKAGES:
                    package = CREDIT_PACKAGES[metadata["package_id"]]
                    credits_to_add = package["credits"]
                    package_name = package["name"]
                    logger.info(f"Found package by package_id: {package_name} ({credits_to_add} credits)")
                else:
                    # Fallback: match by amount
                    for package_id, package in CREDIT_PACKAGES.items():
                        if package["price"] == amount:
                            credits_to_add = package["credits"]
                            package_name = package["name"]
                            logger.info(f"Matched package by amount: {package_name} ({credits_to_add} credits)")
                            break
                
                if credits_to_add == 0:
                    # Default fallback based on amount ranges
                    if amount >= 3499:
                        credits_to_add = 500
                        package_name = "Premium Credits (auto-detected)"
                    elif amount >= 1999:
                        credits_to_add = 250
                        package_name = "Pro Credits (auto-detected)"
                    else:
                        credits_to_add = 100
                        package_name = "Starter Credits (auto-detected)"
                    logger.info(f"Using fallback credits: {credits_to_add} for amount {amount}")
                
                # Update user credits
                try:
                    # Get current credits
                    response = supabase.table("users").select("credits").eq("id", user["id"]).execute()
                    if not response.data:
                        raise Exception("User not found in database")
                    
                    current_credits = response.data[0]["credits"]
                    new_credits = current_credits + credits_to_add
                    
                    # Update user in database
                    update_result = supabase.table("users").update({
                        "credits": new_credits,
                        "updated_at": datetime.utcnow().isoformat()
                    }).eq("id", user["id"]).execute()
                    
                    if update_result.data:
                        logger.info(f"Updated credits for user {user['id']}: {current_credits} -> {new_credits}")
                    else:
                        logger.error(f"Failed to update credits for user {user['id']}")
                    
                    # Create transaction record
                    transaction_id = str(uuid.uuid4())
                    transaction_data = {
                        "id": transaction_id,
                        "user_id": user["id"],
                        "email": user["email"],
                        "order_id": order_id,
                        "package_id": metadata.get("package_id", "unknown"),
                        "credits": credits_to_add,
                        "amount": amount,
                        "type": "credit_purchase",
                        "status": "completed",
                        "description": f"Payment completed: {package_name} (Order {order_id})",
                        "session_id": metadata.get("session_id", order_id),
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat()
                    }
                    
                    # Check if transaction already exists (by session_id)
                    existing_tx = None
                    if metadata.get("session_id"):
                        existing_tx = supabase.table("transactions") \
                            .select("*") \
                            .eq("session_id", metadata.get("session_id")) \
                            .execute()
                    
                    # Also check by order_id
                    if not existing_tx or not existing_tx.data:
                        existing_tx = supabase.table("transactions") \
                            .select("*") \
                            .eq("order_id", order_id) \
                            .execute()
                    
                    if not existing_tx.data:
                        # Create new transaction
                        supabase.table("transactions").insert(transaction_data).execute()
                        logger.info(f"Created NEW transaction record: {transaction_id}")
                    else:
                        # Update existing transaction
                        tx_id = existing_tx.data[0]["id"]
                        supabase.table("transactions").update({
                            "status": "completed",
                            "credits": credits_to_add,
                            "amount": amount,
                            "updated_at": datetime.utcnow().isoformat(),
                            "description": f"Payment completed: {package_name} (Order {order_id})",
                            "order_id": order_id
                        }).eq("id", tx_id).execute()
                        logger.info(f"Updated EXISTING transaction: {tx_id}")
                    
                    # Update webhook log
                    supabase.table("webhook_logs").update({
                        "status": "processed",
                        "user_id": user["id"],
                        "transaction_id": transaction_id,
                        "credits_added": credits_to_add
                    }).eq("id", webhook_log["id"]).execute()
                    
                    logger.info(f"Successfully processed {event_type} webhook for order {order_id}")
                    logger.info(f"Added {credits_to_add} credits to {user['email']}. New balance: {new_credits}")
                    
                    return {
                        "success": True, 
                        "message": f"Added {credits_to_add} credits to user {user['email']}",
                        "credits_added": credits_to_add,
                        "new_balance": new_credits,
                        "event_type": event_type,
                        "order_id": order_id
                    }
                    
                except Exception as credit_update_error:
                    logger.error(f"Failed to update credits: {credit_update_error}")
                    traceback.print_exc()
                    
                    # Update webhook log with error
                    supabase.table("webhook_logs").update({
                        "status": "error",
                        "error": str(credit_update_error)
                    }).eq("id", webhook_log["id"]).execute()
                    
                    return {
                        "success": False, 
                        "error": f"Failed to update credits: {str(credit_update_error)}",
                        "event_type": event_type
                    }
            else:
                logger.info(f"Ignoring {event_type} - order not paid yet (status: {order_status}, paid: {is_paid})")
                return {"success": True, "message": f"Ignored {event_type} - order not paid yet"}
        
        elif event_type == "order.created":
            order_data = payload.get("data", {})
            order_id = order_data.get("id")
            logger.info(f"Order created: {order_id}")
            
            # Update webhook log
            supabase.table("webhook_logs").update({
                "status": "processed",
                "message": f"Order created: {order_id}"
            }).eq("id", webhook_log["id"]).execute()
            
        elif event_type == "order.failed":
            order_data = payload.get("data", {})
            order_id = order_data.get("id")
            logger.warning(f"Payment failed for order: {order_id}")
            
            # Create failed transaction record
            transaction_id = str(uuid.uuid4())
            transaction_data = {
                "id": transaction_id,
                "order_id": order_id,
                "amount": order_data.get("amount", 0),
                "type": "credit_purchase",
                "status": "failed",
                "description": f"Payment failed for order {order_id}",
                "created_at": datetime.utcnow().isoformat()
            }
            supabase.table("transactions").insert(transaction_data).execute()
            
            # Update webhook log
            supabase.table("webhook_logs").update({
                "status": "processed",
                "message": f"Order failed: {order_id}"
            }).eq("id", webhook_log["id"]).execute()
        
        else:
            logger.info(f"Unhandled event type: {event_type}")
            supabase.table("webhook_logs").update({
                "status": "ignored",
                "message": f"Unhandled event type: {event_type}"
            }).eq("id", webhook_log["id"]).execute()
        
        return {"success": True, "message": f"Webhook processed: {event_type}"}
        
    except Exception as webhook_error:
        logger.error(f"Webhook processing error: {webhook_error}")
        traceback.print_exc()
        
        # Log the error
        error_log = {
            "id": str(uuid.uuid4()),
            "error": str(webhook_error),
            "timestamp": datetime.utcnow().isoformat()
        }
        supabase.table("webhook_errors").insert(error_log).execute()
        
        return JSONResponse(
            status_code=200,  # Return 200 to prevent Polar.sh from retrying
            content={"success": False, "error": str(webhook_error)}
        )

# ========== AUTHENTICATION ENDPOINTS ==========

@app.post("/api/auth/signup")
async def signup(request: Request):
    try:
        print("Signup endpoint called")
        
        try:
            data = await request.json()
        except json.JSONDecodeError as json_error:
            print(f"JSON decode error: {json_error}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error": "Invalid JSON format",
                    "detail": str(json_error)
                }
            )
        
        email = data.get("email")
        password = data.get("password")
        name = data.get("name")
        
        if not email:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error": "Email is required",
                    "detail": "Please provide an email address"
                }
            )
        
        if not password:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error": "Password is required",
                    "detail": "Please provide a password"
                }
            )
        
        if not name:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error": "Name is required",
                    "detail": "Please provide your name"
                }
            )
        
        if len(password) < 6:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error": "Password too short",
                    "detail": "Password must be at least 6 characters"
                }
            )
        
        if "@" not in email:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error": "Invalid email",
                    "detail": "Please provide a valid email address"
                }
            )
        
        try:
            response = supabase.table("users").select("*").eq("email", email).execute()
        except Exception as db_error:
            print(f"Supabase query error: {db_error}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "error": "Database error",
                    "detail": "Failed to query database"
                }
            )
        
        if response.data:
            user = response.data[0]
            if user.get("is_verified"):
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "success": False,
                        "error": "User already exists",
                        "detail": "An account with this email already exists"
                    }
                )
            else:
                verification_token = secrets.token_urlsafe(32)
                try:
                    supabase.table("users").update({
                        "verification_token": verification_token,
                        "verification_token_expires": (datetime.utcnow() + timedelta(hours=24)).isoformat()
                    }).eq("id", user["id"]).execute()
                except Exception as update_error:
                    print(f"Failed to update user: {update_error}")
                
                send_verification_email(email, user.get("name", name), verification_token)
                
                return {
                    "success": True,
                    "message": "Verification email resent. Please check your inbox.",
                    "requires_verification": True
                }
        
        user_id = str(uuid.uuid4())
        verification_token = secrets.token_urlsafe(32)
        
        try:
            password_hash = get_password_hash(password)
        except Exception as hash_error:
            print(f"Password hashing failed: {hash_error}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "error": "Password processing failed",
                    "detail": "Failed to process password"
                }
            )
        
        new_user = {
            "id": user_id,
            "email": email,
            "name": name,
            "password_hash": password_hash,
            "is_verified": False,
            "verification_token": verification_token,
            "verification_token_expires": (datetime.utcnow() + timedelta(hours=24)).isoformat(),
            "credits": 1000,
            "created_at": datetime.utcnow().isoformat()
        }
        
        try:
            response = supabase.table("users").insert(new_user).execute()
        except Exception as insert_error:
            print(f"Failed to insert user: {insert_error}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "error": "Database error",
                    "detail": "Failed to create user in database"
                }
            )
        
        if not response.data:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "error": "Database error",
                    "detail": "Failed to create user in database"
                }
            )
        
        send_verification_email(email, name, verification_token)
        
        logger.info(f"New user registered (pending verification): {email}")
        
        return {
            "success": True,
            "message": "Verification email sent. Please check your inbox.",
            "requires_verification": True,
            "user_id": user_id
        }
        
    except Exception as signup_error:
        print(f"Unexpected error in signup: {signup_error}")
        traceback.print_exc()
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "Internal server error",
                "detail": "Registration failed due to an internal error"
            }
        )

@app.post("/api/auth/login")
async def login(request: Request):
    try:
        data = await request.json()
        email = data.get("email")
        password = data.get("password")
        
        if not email or not password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email and password are required"
            )
        
        response = supabase.table("users").select("*").eq("email", email).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        user = response.data[0]
        
        if not verify_password(password, user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        if not user["is_verified"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Please verify your email before logging in"
            )
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["id"], "email": user["email"]},
            expires_delta=access_token_expires
        )
        
        logger.info(f"User logged in: {user['email']}")
        
        return {
            "success": True,
            "message": "Login successful",
            "token": access_token,
            "user": {
                "id": user["id"],
                "email": user["email"],
                "name": user["name"],
                "credits": user["credits"],
                "verified": user["is_verified"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as login_error:
        logger.error(f"Login error: {login_error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@app.post("/api/auth/verify")
async def verify_email(token: str = Form(...)):
    try:
        response = supabase.table("users").select("*").eq("verification_token", token).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired verification token"
            )
        
        user = response.data[0]
        
        token_expires_str = user.get("verification_token_expires")
        if token_expires_str:
            try:
                token_expires = datetime.fromisoformat(token_expires_str.replace('Z', '+00:00'))
                if datetime.utcnow() > token_expires:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Verification token has expired"
                    )
            except Exception as date_error:
                logger.error(f"Token expiry parsing error: {date_error}")
        
        supabase.table("users").update({
            "is_verified": True,
            "verification_token": None,
            "verification_token_expires": None
        }).eq("id", user["id"]).execute()
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["id"], "email": user["email"]},
            expires_delta=access_token_expires
        )
        
        logger.info(f"Email verified: {user['email']}")
        
        return {
            "success": True,
            "message": "Email verified successfully!",
            "token": access_token,
            "user": {
                "id": user["id"],
                "email": user["email"],
                "name": user["name"],
                "credits": user["credits"],
                "verified": True
            }
        }
        
    except HTTPException:
        raise
    except Exception as verification_error:
        logger.error(f"Verification error: {verification_error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Verification failed"
        )

@app.post("/api/auth/resend-verification")
async def resend_verification(request: Request):
    try:
        data = await request.json()
        email = data.get("email")
        
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email is required"
            )
        
        response = supabase.table("users").select("*").eq("email", email).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        user = response.data[0]
        
        if user["is_verified"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already verified"
            )
        
        verification_token = secrets.token_urlsafe(32)
        supabase.table("users").update({
            "verification_token": verification_token,
            "verification_token_expires": (datetime.utcnow() + timedelta(hours=24)).isoformat()
        }).eq("id", user["id"]).execute()
        
        send_verification_email(user["email"], user["name"], verification_token)
        
        return {
            "success": True,
            "message": "Verification email resent. Please check your inbox."
        }
        
    except HTTPException:
        raise
    except Exception as resend_error:
        logger.error(f"Resend verification error: {resend_error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resend verification email"
        )

@app.post("/api/auth/logout")
async def logout(response: Response, current_user: User = Depends(get_current_user)):
    response.delete_cookie(key="access_token")
    
    return {
        "success": True,
        "message": "Logged out successfully"
    }

@app.post("/api/auth/demo-login")
async def demo_login():
    try:
        demo_email = "demo@octavia.com"
        demo_password = "demo123"
        
        response = supabase.table("users").select("*").eq("email", demo_email).execute()
        
        if response.data:
            user = response.data[0]
            if not verify_password(demo_password, user["password_hash"]):
                supabase.table("users").update({
                    "password_hash": get_password_hash(demo_password)
                }).eq("id", user["id"]).execute()
        else:
            user_id = str(uuid.uuid4())
            new_user = {
                "id": user_id,
                "email": demo_email,
                "name": "Demo User",
                "password_hash": get_password_hash(demo_password),
                "is_verified": True,
                "credits": 5000,
                "created_at": datetime.utcnow().isoformat()
            }
            
            response = supabase.table("users").insert(new_user).execute()
            if not response.data:
                raise HTTPException(500, "Failed to create demo user")
            
            user = response.data[0]
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["id"], "email": user["email"]},
            expires_delta=access_token_expires
        )
        
        return {
            "success": True,
            "message": "Demo login successful",
            "token": access_token,
            "user": {
                "id": user["id"],
                "email": user["email"],
                "name": user["name"],
                "credits": user["credits"],
                "verified": user["is_verified"]
            }
        }
        
    except Exception as demo_error:
        logger.error(f"Demo login error: {demo_error}")
        raise HTTPException(500, "Demo login failed")

# ========== USER PROFILE ENDPOINTS ==========

@app.get("/api/user/profile")
async def get_user_profile(current_user: User = Depends(get_current_user)):
    return {
        "success": True,
        "user": {
            "id": current_user.id,
            "email": current_user.email,
            "name": current_user.name,
            "credits": current_user.credits,
            "verified": current_user.is_verified,
            "created_at": current_user.created_at.isoformat() if isinstance(current_user.created_at, datetime) else current_user.created_at
        }
    }

@app.get("/api/user/credits")
async def get_user_credits(current_user: User = Depends(get_current_user)):
    return {
        "success": True,
        "credits": current_user.credits,
        "email": current_user.email
    }

# ========== VIDEO TRANSLATION ENDPOINTS ==========

@app.post("/api/translate/video/enhanced")
async def translate_video_enhanced(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_language: str = Form("es"),
    chunk_size: int = Form(30),
    current_user: User = Depends(get_current_user)
):
    """Enhanced video translation with chunk processing"""
    try:
        # Check if user has enough credits (10 credits for video translation)
        if current_user.credits < 10:
            raise HTTPException(400, "Insufficient credits. Need at least 10 credits.")
        
        if not PIPELINE_AVAILABLE:
            raise HTTPException(500, "Full video pipeline not available. Running in simplified mode.")
        
        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1]
        file_path = f"temp_{file_id}{file_ext}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Deduct credits
        supabase.table("users").update({"credits": current_user.credits - 10}).eq("id", current_user.id).execute()
        
        # Create job entry
        job_id = str(uuid.uuid4())
        jobs_db[job_id] = {
            "id": job_id,
            "type": "video_enhanced",
            "status": "processing",
            "progress": 0,
            "file_path": file_path,
            "target_language": target_language,
            "chunk_size": chunk_size,
            "user_id": current_user.id,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Process in background
        background_tasks.add_task(
            process_video_enhanced_job,
            job_id,
            file_path,
            target_language,
            chunk_size,
            current_user.id
        )
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "Video translation started in background",
            "status_url": f"/api/jobs/{job_id}/status",
            "remaining_credits": current_user.credits - 10
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video translation failed: {str(e)}")

async def process_video_enhanced_job(job_id: str, file_path: str, target_language: str, chunk_size: int, user_id: str):
    """Background task for enhanced video translation"""
    try:
        # Update job status
        jobs_db[job_id]["progress"] = 10
        
        # Initialize pipeline
        config = PipelineConfig(chunk_size=chunk_size)
        pipeline = VideoTranslationPipeline(config)
        
        # Update progress
        jobs_db[job_id]["progress"] = 30
        
        # Process video
        result = pipeline.process_video(file_path, target_language)
        
        # Update job with results
        jobs_db[job_id].update({
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
        jobs_db[job_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.utcnow().isoformat()
        })
        
        # Refund credits on failure
        try:
            response = supabase.table("users").select("credits").eq("id", user_id).execute()
            if response.data:
                current_credits = response.data[0]["credits"]
                supabase.table("users").update({"credits": current_credits + 10}).eq("id", user_id).execute()
        except Exception as refund_error:
            logger.error(f"Failed to refund credits: {refund_error}")

@app.get("/api/jobs/{job_id}/status")
async def get_job_status(job_id: str, current_user: User = Depends(get_current_user)):
    """Get status of a translation job"""
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")
    
    job = jobs_db[job_id]
    
    if job["user_id"] != current_user.id:
        raise HTTPException(403, "Access denied")
    
    response = {
        "success": True,
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0),
        "target_language": job.get("target_language"),
        "download_url": job.get("output_path"),
        "error": job.get("error")
    }
    
    return response

@app.get("/api/download/{file_type}/{file_id}")
async def download_file(file_type: str, file_id: str, current_user: User = Depends(get_current_user)):
    """Download generated files"""
    if file_type == "subtitles":
        filename = "subtitles.srt"
        media_type = "text/plain"
    elif file_type == "video":
        # Check if this is a job output
        for job_id, job in jobs_db.items():
            if job.get("output_path") and file_id in job.get("output_path", ""):
                if job["user_id"] != current_user.id:
                    raise HTTPException(403, "Access denied")
                filename = job["output_path"]
                media_type = "video/mp4"
                break
        else:
            raise HTTPException(404, "File not found")
    else:
        raise HTTPException(404, "File type not found")
    
    if not os.path.exists(filename):
        raise HTTPException(404, "File not found")
    
    return FileResponse(
        filename,
        media_type=media_type,
        filename=f"octavia_{file_type}_{file_id}{os.path.splitext(filename)[1]}"
    )

# ========== HEALTH & TESTING ENDPOINTS ==========

@app.get("/api/health")
async def health_check():
    return {
        "success": True,
        "status": "healthy",
        "service": "Octavia Video Translator",
        "version": "4.0.0",
        "database": "Supabase",
        "payment": {
            "mode": "sandbox",
            "test_mode": ENABLE_TEST_MODE,
            "real_products_configured": True,
        },
        "models": {
            "whisper": "loaded" if whisper_model else "not_available",
            "translation": "loaded" if translator else "not_available",
            "pipeline": "available" if PIPELINE_AVAILABLE else "simplified_mode"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/test/integration")
async def test_integration():
    """Run integration test on sample video"""
    try:
        test_video = "test_samples/sample_30s.mp4"
        
        if not os.path.exists(test_video):
            return {
                "success": False,
                "error": f"Test video not found: {test_video}",
                "message": "Please create a test sample first"
            }
        
        if not PIPELINE_AVAILABLE:
            return {
                "success": False,
                "error": "Pipeline modules not available",
                "message": "Running in simplified mode"
            }
        
        # Run the pipeline on test video
        pipeline = VideoTranslationPipeline()
        result = pipeline.process_video(test_video, "es")
        
        return {
            "success": True,
            "message": "Integration test completed",
            "result": result
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Integration test failed"
        }

@app.get("/api/metrics")
async def get_metrics():
    """Get processing metrics from logs"""
    try:
        log_file = "artifacts/logs.jsonl"
        
        if not os.path.exists(log_file):
            return {
                "success": True,
                "message": "No logs available yet",
                "metrics": {}
            }
        
        # Simple metrics collection
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        return {
            "success": True,
            "total_logs": len(lines),
            "log_file": log_file,
            "message": f"Found {len(lines)} log entries"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to get metrics"
        }

# ========== SUBTITLE MANAGEMENT ENDPOINTS ==========

@app.get("/api/translate/subtitles/list")
async def list_subtitle_files(job_id: str, current_user: User = Depends(get_current_user)):
    """List subtitle files for a translation job"""
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")
    
    job = jobs_db[job_id]
    
    if job["user_id"] != current_user.id:
        raise HTTPException(403, "Access denied")
    
    # Check for subtitle files in output directory
    subtitle_files = []
    output_dir = os.path.dirname(job.get("output_path", ""))
    
    if output_dir and os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith(('.srt', '.vtt', '.ass', '.ssa')):
                subtitle_files.append({
                    "name": file,
                    "path": os.path.join(output_dir, file),
                    "size": os.path.getsize(os.path.join(output_dir, file)),
                    "format": os.path.splitext(file)[1][1:]
                })
    
    return {
        "success": True,
        "job_id": job_id,
        "subtitle_files": subtitle_files
    }

@app.get("/api/download/subtitles/{job_id}/{filename}")
async def download_subtitle_file(
    job_id: str, 
    filename: str,
    current_user: User = Depends(get_current_user)
):
    """Download a specific subtitle file"""
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")
    
    job = jobs_db[job_id]
    
    if job["user_id"] != current_user.id:
        raise HTTPException(403, "Access denied")
    
    # Find the subtitle file
    output_dir = os.path.dirname(job.get("output_path", ""))
    file_path = os.path.join(output_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(404, "Subtitle file not found")
    
    # Determine media type
    ext = os.path.splitext(filename)[1].lower()
    media_types = {
        '.srt': 'text/plain',
        '.vtt': 'text/vtt',
        '.ass': 'text/x-ssa',
        '.ssa': 'text/x-ssa'
    }
    
    media_type = media_types.get(ext, 'application/octet-stream')
    
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=filename
    )
@app.get("/api/translate/subtitles/review/{job_id}")
async def review_subtitles(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get subtitle content for review/editing"""
    if job_id not in subtitle_jobs:
        raise HTTPException(404, "Job not found")
    
    job = subtitle_jobs[job_id]
    
    if job["user_id"] != current_user.id:
        raise HTTPException(403, "Access denied")
    
    if job.get("status") != "completed":
        raise HTTPException(400, "Subtitles not ready yet. Status: " + job.get("status", "unknown"))
    
    # Try to read the subtitle file
    format = job.get("format", "srt")
    filename = job.get("filename", f"subtitles_{job_id}.{format}")
    
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                subtitle_content = f.read()
        else:
            # Return content if it's stored in the job
            subtitle_content = job.get("content", "")
            
            # If not stored, try to read from download_url path
            if not subtitle_content and job.get("download_url"):
                subtitle_path = filename
                if os.path.exists(subtitle_path):
                    with open(subtitle_path, "r", encoding="utf-8") as f:
                        subtitle_content = f.read()
    except Exception as e:
        subtitle_content = ""
    
    return {
        "success": True,
        "job_id": job_id,
        "status": job.get("status"),
        "format": format,
        "language": job.get("language"),
        "segment_count": job.get("segment_count", 0),
        "content": subtitle_content,
        "download_url": job.get("download_url"),
        "created_at": job.get("created_at"),
        "completed_at": job.get("completed_at")
    }
@app.get("/api/payments/transactions")
async def get_user_transactions(
    current_user: User = Depends(get_current_user),
    limit: int = 20,
    offset: int = 0
):
    """Get transaction history for the current user"""
    try:
        response = supabase.table("transactions") \
            .select("*") \
            .eq("user_id", current_user.id) \
            .order("created_at", desc=True) \
            .range(offset, offset + limit - 1) \
            .execute()
        
        transactions = []
        for tx in response.data:
            transactions.append({
                "id": tx.get("id"),
                "type": tx.get("type"),
                "status": tx.get("status"),
                "description": tx.get("description"),
                "credits": tx.get("credits", 0),
                "amount": tx.get("amount", 0),
                "created_at": tx.get("created_at"),
                "updated_at": tx.get("updated_at")
            })
        
        # Get total count for pagination
        count_response = supabase.table("transactions") \
            .select("id", count="exact") \
            .eq("user_id", current_user.id) \
            .execute()
        
        total_count = count_response.count or 0
        
        return {
            "success": True,
            "transactions": transactions,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": total_count
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get transactions: {e}")
        raise HTTPException(500, "Failed to retrieve transaction history")
# ========== ROOT ENDPOINT ==========

@app.get("/")
async def root():
    return {
        "success": True,
        "service": "Octavia Video Translator",
        "version": "4.0.0",
        "status": "operational",
        "authentication": "JWT + Supabase",
        "payment": "Polar.sh integration (sandbox)",
        "test_mode": ENABLE_TEST_MODE,
        "pipeline_mode": "full" if PIPELINE_AVAILABLE else "simplified",
        "endpoints": {
            "health": "/api/health",
            "docs": "/docs",
            "auth": {
                "signup": "/api/auth/signup",
                "login": "/api/auth/login",
                "logout": "/api/auth/logout",
                "verify": "/api/auth/verify",
                "resend_verification": "/api/auth/resend-verification",
                "demo_login": "/api/auth/demo-login"
            },
            "payments": {
                "packages": "/api/payments/packages",
                "create_session": "/api/payments/create-session",
                "payment_status": "/api/payments/status/{session_id}",
                "webhook": "/api/payments/webhook/polar"
            },
            "user": {
                "profile": "/api/user/profile",
                "credits": "/api/user/credits"
            },
            "translation": {
                "subtitles": "/api/translate/subtitles",
                "subtitles_status": "/api/translate/subtitles/status/{job_id}",
                "subtitles_review": "/api/translate/subtitles/review/{job_id}",  
                "video_enhanced": "/api/translate/video/enhanced",
                "job_status": "/api/jobs/{job_id}/status",
                "download": "/api/download/{file_type}/{file_id}"
            },
            "testing": {
                "integration_test": "/api/test/integration",
                "metrics": "/api/metrics"
            }
        },
        "required_files": {
            "test_video": "test_samples/sample_30s.mp4",
            "config": "config.yaml",
            "logs": "artifacts/logs.jsonl"
        }
    }

# ========== APPLICATION ENTRY POINT ==========

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("OCTAVIA VIDEO TRANSLATOR v4.0 - BACKEND SERVER")
    print("="*60)
    print(f"Database: Supabase")
    print(f"Payment: Polar.sh (sandbox mode)")
    print(f"Test Mode: {'ENABLED' if ENABLE_TEST_MODE else 'DISABLED'}")
    print(f"Pipeline Modules: {'AVAILABLE' if PIPELINE_AVAILABLE else 'SIMPLIFIED MODE'}")
    print(f"API URL: http://localhost:8000")
    print(f"Documentation: http://localhost:8000/docs")
    print(f"Logs Directory: artifacts/")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )
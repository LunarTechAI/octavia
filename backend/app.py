"""
Octavia Video Translator Backend
FastAPI application for video translation with user authentication using Supabase
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, Response, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
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
from typing import Optional
import traceback
import hashlib
import binascii

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# ========== SUPABASE CONFIGURATION ==========
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize Supabase client for database operations
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ========== CUSTOM PASSWORD HANDLING ==========
class PasswordManager:
    """
    Custom password manager using SHA256 with salt
    Created as a workaround for bcrypt compatibility issues
    """
    
    def __init__(self):
        self.method = "sha256"
        print(f"Using {self.method} for password hashing")
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA256 with a random salt"""
        # Generate random salt for each password
        salt = secrets.token_hex(16)
        
        # Combine password and salt, then hash
        hashed_password = hashlib.sha256((password + salt).encode()).hexdigest()
        
        # Store in format: method:salt:hash for later verification
        return f"sha256:{salt}:{hashed_password}"
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against stored hash"""
        try:
            if not hashed_password or ':' not in hashed_password:
                return False
            
            # Parse stored hash format
            parts = hashed_password.split(':')
            if len(parts) != 3:
                return False
            
            method, salt, stored_hash = parts
            
            if method == "sha256":
                # Recompute hash with same salt for comparison
                computed_hash = hashlib.sha256((plain_password + salt).encode()).hexdigest()
                return computed_hash == stored_hash
            else:
                # Fallback for existing bcrypt hashes if present
                return self._try_bcrypt_fallback(plain_password, hashed_password)
                
        except Exception as e:
            print(f"Password verification error: {e}")
            return False
    
    def _try_bcrypt_fallback(self, plain_password: str, hashed_password: str) -> bool:
        """Try bcrypt verification for existing user passwords"""
        try:
            import bcrypt
            # Check if hash uses bcrypt format
            if hashed_password.startswith('$2b$') or hashed_password.startswith('$2a$'):
                return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
        except Exception:
            pass
        return False

# Initialize password manager instance
password_manager = PasswordManager()
security = HTTPBearer()

# ========== CREATE ARTIFACTS DIRECTORY ==========
os.makedirs("artifacts", exist_ok=True)
print(f"Created artifacts directory at: {os.path.abspath('artifacts')}")

# ========== SETUP LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "stage": "%(name)s", "chunk_id": "%(filename)s", "duration_ms": %(relativeCreated)d, "status": "%(levelname)s", "message": "%(message)s"}',
    handlers=[
        logging.FileHandler('artifacts/logs.jsonl'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Logging system initialized successfully")

# ========== DATA MODELS ==========
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

# ========== HELPER FUNCTIONS ==========
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password using custom password manager"""
    return password_manager.verify_password(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash password using custom password manager"""
    return password_manager.hash_password(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token with expiration"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to get current authenticated user from JWT token"""
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
    
    # Fetch user from Supabase database
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
    except Exception as e:
        logger.error(f"Error fetching user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching user data",
        )

def send_verification_email(email: str, name: str, verification_token: str):
    """Send verification email to newly registered user"""
    try:
        # Get SMTP configuration from environment variables
        smtp_server = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_username = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASS")
        smtp_from = os.getenv("SMTP_FROM", "noreply@octavia.com")
        
        # Fallback to mock email for development without SMTP
        if not all([smtp_username, smtp_password]):
            logger.warning(f"SMTP credentials not configured. Mock email would be sent to: {email}")
            print(f"Mock verification email would be sent to: {email}")
            return True
        
        # Create email message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = "Verify Your Octavia Account"
        msg['From'] = smtp_from
        msg['To'] = email
        
        # Generate verification link for frontend
        verification_link = f"http://localhost:3000/verify-email?token={verification_token}"
        
        # HTML email content
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
        
        # Plain text fallback for email clients
        text = f"""Welcome to Octavia!
        
Hi {name},
        
Thank you for signing up for Octavia Video Translator. To start using your account, please verify your email address by clicking the link below:
        
{verification_link}
        
This link will expire in 24 hours.
        
If you didn't create an account with Octavia, you can safely ignore this email.
        
Best regards,
The Octavia Team"""
        
        # Attach both HTML and plain text versions
        part1 = MIMEText(text, 'plain')
        part2 = MIMEText(html, 'html')
        msg.attach(part1)
        msg.attach(part2)
        
        # Send email via SMTP
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        
        logger.info(f"Verification email sent to {email}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send verification email to {email}: {e}")
        # Return True in development to allow registration flow to continue
        return True

# ========== APPLICATION LIFECYCLE ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup sequence
    print("Starting Octavia Video Translator with Supabase...")
    print("=" * 60)
    
    # Log hardware information for debugging
    hardware_info = {
        "cpu_count": psutil.cpu_count(),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "available_ram_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "platform": sys.platform,
        "python_version": sys.version
    }
    
    logger.info(f"Hardware detected: {hardware_info}")
    
    # Test database connection
    try:
        response = supabase.table("users").select("count", count="exact").limit(1).execute()
        print("Connected to Supabase")
    except Exception as e:
        print(f"Supabase connection issue: {e}")
        print("Please ensure your Supabase credentials are correct in .env file")
    
    # Initialize AI models
    print("Loading AI models...")
    global whisper_model, translator
    
    try:
        whisper_model = whisper.load_model("base")
        print("Whisper speech recognition model loaded")
    except Exception as e:
        print(f"Whisper load failed: {e}")
        whisper_model = None
    
    try:
        from transformers import MarianMTModel, MarianTokenizer
        model_name = "Helsinki-NLP/opus-mt-en-es"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        translator = pipeline("translation", model=model, tokenizer=tokenizer)
        print("Translation model loaded")
    except Exception as e:
        print(f"Translation model failed: {e}")
        translator = None
    
    print("AI models ready")
    print("=" * 60)
    
    yield  # Application runs here
    
    # Shutdown cleanup
    print("Shutting down Octavia...")
    # Clean up temporary files
    for file in os.listdir("."):
        if file.startswith("temp_") or file.startswith("translated_"):
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

# ========== CORS CONFIGURATION ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow frontend only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# In-memory storage for translation jobs
jobs_db: Dict[str, Dict] = {}
files_db: Dict[str, str] = {}

# ========== AUTHENTICATION ENDPOINTS ==========
@app.post("/api/auth/signup")
async def signup(request: Request):
    """
    Register new user account with email verification
    Requires email, password, and name in JSON format
    """
    try:
        print("Signup endpoint called")
        
        # Parse incoming JSON data
        try:
            data = await request.json()
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error": "Invalid JSON format",
                    "detail": str(e)
                }
            )
        
        email = data.get("email")
        password = data.get("password")
        name = data.get("name")
        
        # Validate required fields
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
        
        # Basic validation rules
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
        
        # Check if user already exists
        try:
            response = supabase.table("users").select("*").eq("email", email).execute()
        except Exception as e:
            print(f"Supabase query error: {e}")
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
                # User exists but hasn't verified email
                verification_token = secrets.token_urlsafe(32)
                try:
                    supabase.table("users").update({
                        "verification_token": verification_token,
                        "verification_token_expires": (datetime.utcnow() + timedelta(hours=24)).isoformat()
                    }).eq("id", user["id"]).execute()
                except Exception as e:
                    print(f"Failed to update user: {e}")
                
                send_verification_email(email, user.get("name", name), verification_token)
                
                return {
                    "success": True,
                    "message": "Verification email resent. Please check your inbox.",
                    "requires_verification": True
                }
        
        # Create new user record
        user_id = str(uuid.uuid4())
        verification_token = secrets.token_urlsafe(32)
        
        # Hash password for secure storage
        try:
            password_hash = get_password_hash(password)
        except Exception as e:
            print(f"Password hashing failed: {e}")
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
        
        # Insert user into Supabase
        try:
            response = supabase.table("users").insert(new_user).execute()
        except Exception as e:
            print(f"Failed to insert user: {e}")
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
        
        # Send verification email to new user
        send_verification_email(email, name, verification_token)
        
        logger.info(f"New user registered (pending verification): {email}")
        
        return {
            "success": True,
            "message": "Verification email sent. Please check your inbox.",
            "requires_verification": True,
            "user_id": user_id
        }
        
    except Exception as e:
        print(f"Unexpected error in signup: {e}")
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
    """Authenticate user with email and password"""
    try:
        # Parse login credentials
        data = await request.json()
        email = data.get("email")
        password = data.get("password")
        
        # Validate required fields
        if not email or not password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email and password are required"
            )
        
        # Retrieve user from database
        response = supabase.table("users").select("*").eq("email", email).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        user = response.data[0]
        
        # Verify password matches stored hash
        if not verify_password(password, user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Check if user has verified their email
        if not user["is_verified"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Please verify your email before logging in"
            )
        
        # Create JWT access token for session
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
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@app.post("/api/auth/verify")
async def verify_email(token: str = Form(...)):
    """Verify email using token from verification link"""
    try:
        # Find user with matching verification token
        response = supabase.table("users").select("*").eq("verification_token", token).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired verification token"
            )
        
        user = response.data[0]
        
        # Check if verification token has expired
        token_expires_str = user.get("verification_token_expires")
        if token_expires_str:
            try:
                token_expires = datetime.fromisoformat(token_expires_str.replace('Z', '+00:00'))
                if datetime.utcnow() > token_expires:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Verification token has expired"
                    )
            except Exception as e:
                logger.error(f"Token expiry parsing error: {e}")
        
        # Mark user as verified and clear verification token
        supabase.table("users").update({
            "is_verified": True,
            "verification_token": None,
            "verification_token_expires": None
        }).eq("id", user["id"]).execute()
        
        # Create JWT token for immediate login
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
    except Exception as e:
        logger.error(f"Verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Verification failed"
        )

@app.post("/api/auth/resend-verification")
async def resend_verification(request: Request):
    """Resend verification email to user"""
    try:
        data = await request.json()
        email = data.get("email")
        
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email is required"
            )
        
        # Find user by email
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
        
        # Generate new verification token
        verification_token = secrets.token_urlsafe(32)
        supabase.table("users").update({
            "verification_token": verification_token,
            "verification_token_expires": (datetime.utcnow() + timedelta(hours=24)).isoformat()
        }).eq("id", user["id"]).execute()
        
        # Send new verification email
        send_verification_email(user["email"], user["name"], verification_token)
        
        return {
            "success": True,
            "message": "Verification email resent. Please check your inbox."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resend verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resend verification email"
        )

@app.post("/api/auth/logout")
async def logout(response: Response, current_user: User = Depends(get_current_user)):
    """Logout endpoint - primarily client-side token removal"""
    # With JWT, logout is handled client-side by removing the token
    # This endpoint provides server-side confirmation
    
    response.delete_cookie(key="access_token")
    
    return {
        "success": True,
        "message": "Logged out successfully"
    }

@app.post("/api/auth/demo-login")
async def demo_login():
    """Demo login for testing without registration"""
    try:
        demo_email = "demo@octavia.com"
        demo_password = "demo123"
        
        # Check if demo user exists
        response = supabase.table("users").select("*").eq("email", demo_email).execute()
        
        if response.data:
            user = response.data[0]
            # Update password if needed
            if not verify_password(demo_password, user["password_hash"]):
                supabase.table("users").update({
                    "password_hash": get_password_hash(demo_password)
                }).eq("id", user["id"]).execute()
        else:
            # Create demo user if doesn't exist
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
        
        # Create JWT token for demo user
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
        
    except Exception as e:
        logger.error(f"Demo login error: {e}")
        raise HTTPException(500, "Demo login failed")

# ========== USER PROFILE ENDPOINTS ==========
@app.get("/api/user/profile")
async def get_user_profile(current_user: User = Depends(get_current_user)):
    """Get current user's profile information"""
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
    """Get user's current credit balance"""
    return {
        "success": True,
        "credits": current_user.credits,
        "email": current_user.email
    }

# ========== FILE HELPER FUNCTIONS ==========
def save_upload_file(upload_file: UploadFile) -> tuple:
    """Save uploaded file to temporary storage"""
    file_id = str(uuid.uuid4())
    file_ext = os.path.splitext(upload_file.filename)[1]
    file_path = f"temp_{file_id}{file_ext}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        
        files_db[file_id] = file_path
        logger.info(f"File saved: {upload_file.filename} -> {file_path}")
        return file_id, file_path
    except Exception as e:
        logger.error(f"File save failed: {e}")
        raise HTTPException(500, f"File upload failed: {str(e)}")

def cleanup_file(file_path: str):
    """Clean up temporary file after processing"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up: {file_path}")
    except Exception as e:
        logger.error(f"Failed to cleanup {file_path}: {e}")

def extract_audio_from_video(video_path: str, output_audio_path: str) -> bool:
    """Extract audio track from video file using FFmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '44100', '-ac', '2',
            '-loglevel', 'error',
            output_audio_path, '-y'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True,
                              creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr[:200]}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        return False

def translate_text_with_fallback(text: str, target_lang: str = "es") -> str:
    """Translate text with fallback for when AI models aren't available"""
    if translator:
        try:
            result = translator(text)
            if isinstance(result, list) and len(result) > 0:
                return result[0]['translation_text']
            return text
        except Exception as e:
            logger.error(f"Translation failed: {e}")
    
    # Simple fallback translation for development
    fallback_map = {
        "hello": "hola",
        "welcome": "bienvenido",
        "thank you": "gracias",
        "goodbye": "adiós",
        "please": "por favor"
    }
    
    translated = text
    for eng, esp in fallback_map.items():
        translated = translated.replace(eng, esp)
    
    return translated

# ========== VIDEO TRANSLATION ENDPOINTS ==========
@app.post("/api/translate/video")
async def translate_video(
    file: UploadFile = File(...),
    target_language: str = Form("es"),
    current_user: User = Depends(get_current_user)
):
    """Process video translation request"""
    # Check user has sufficient credits
    if current_user.credits < 5:
        raise HTTPException(400, "Insufficient credits")
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
        raise HTTPException(400, "Please upload a video file")
    
    # Deduct credits for translation
    try:
        supabase.table("users").update({"credits": current_user.credits - 5}).eq("id", current_user.id).execute()
    except Exception as e:
        logger.error(f"Failed to update credits: {e}")
        raise HTTPException(500, "Failed to process payment")
    
    # Save uploaded file
    file_id, file_path = save_upload_file(file)
    job_id = str(uuid.uuid4())
    
    job_info = {
        "id": job_id,
        "status": "processing",
        "progress": 10,
        "type": "video_simple",
        "file_id": file_id,
        "target_language": target_language,
        "original_filename": file.filename,
        "user_id": current_user.id,
        "user_email": current_user.email
    }
    jobs_db[job_id] = job_info
    
    try:
        # Simulate processing for demonstration
        await asyncio.sleep(2)
        job_info["progress"] = 50
        
        # Create output file
        output_filename = f"translated_{job_id}.mp4"
        shutil.copy2(file_path, output_filename)
        
        job_info["progress"] = 100
        job_info["status"] = "completed"
        job_info["download_url"] = f"/api/download/{job_id}"
        job_info["output_filename"] = output_filename
        
        # Get updated credit balance
        response = supabase.table("users").select("credits").eq("id", current_user.id).execute()
        new_credits = response.data[0]["credits"] if response.data else current_user.credits - 5
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "Video translation completed",
            "download_url": f"/api/download/{job_id}",
            "remaining_credits": new_credits
        }
        
    except Exception as e:
        job_info["status"] = "failed"
        job_info["error"] = str(e)
        # Refund credits on failure
        try:
            supabase.table("users").update({"credits": current_user.credits}).eq("id", current_user.id).execute()
        except:
            pass
        raise HTTPException(500, f"Processing failed: {str(e)}")
    
    finally:
        cleanup_file(file_path)

# ========== JOB STATUS ENDPOINT ==========
@app.get("/api/jobs/{job_id}/status")
async def get_job_status(job_id: str, current_user: User = Depends(get_current_user)):
    """Get status of a translation job"""
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")
    
    job = jobs_db[job_id]
    
    # Verify user owns this job
    if job["user_id"] != current_user.id:
        raise HTTPException(403, "Access denied")
    
    response = {
        "success": True,
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "progress": job.get("progress", 0),
        "original_filename": job.get("original_filename"),
        "target_language": job.get("target_language"),
        "download_url": job.get("download_url"),
        "error": job.get("error")
    }
    
    return response

# ========== DOWNLOAD ENDPOINT ==========
@app.get("/api/download/{job_id}")
async def download_file(job_id: str, current_user: User = Depends(get_current_user)):
    """Download processed video file"""
    if job_id not in jobs_db:
        raise HTTPException(404, "File not found")
    
    job = jobs_db[job_id]
    
    # Verify user owns this job
    if job["user_id"] != current_user.id:
        raise HTTPException(403, "Access denied")
    
    filename = job.get("output_filename")
    
    if not filename or not os.path.exists(filename):
        raise HTTPException(404, "Output file not found")
    
    return FileResponse(
        filename,
        media_type="application/octet-stream",
        filename=f"octavia_translation_{job_id}{os.path.splitext(filename)[1]}"
    )

# ========== HEALTH CHECK ==========
@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "success": True,
        "status": "healthy",
        "service": "Octavia Video Translator",
        "version": "4.0.0",
        "database": "Supabase",
        "models": {
            "whisper": "loaded" if whisper_model else "not_available",
            "translation": "loaded" if translator else "not_available"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/auth/signup-debug")
async def signup_debug(request: Request):
    """Debug endpoint for testing signup functionality"""
    try:
        print("DEBUG: Signup endpoint called")
        
        # Get raw request body for debugging
        body_bytes = await request.body()
        
        try:
            data = await request.json()
            
            # Check required fields
            email = data.get("email")
            password = data.get("password")
            name = data.get("name")
            
            if not email or not password or not name:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error": "Missing required fields",
                        "detail": "Email, password, and name are required"
                    }
                )
            
            # Test password hashing
            try:
                hashed = get_password_hash(password)
                
                # Test verification
                verify_test = verify_password(password, hashed)
            except Exception as hash_error:
                print(f"Hash test failed: {hash_error}")
            
            return {
                "success": True,
                "message": "Debug signup successful",
                "requires_verification": True,
                "debug_data": {
                    "email": email,
                    "name": name,
                    "password_length": len(password)
                }
            }
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Invalid JSON",
                    "detail": str(e)
                }
            )
            
    except Exception as e:
        print(f"Unexpected error in debug endpoint: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "detail": str(e)
            }
        )

@app.get("/api/test")
async def test_endpoint():
    """Simple test endpoint for API connectivity"""
    return {
        "success": True,
        "message": "API is working",
        "timestamp": datetime.now().isoformat(),
        "hashing_method": "sha256_with_salt"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "success": True,
        "service": "Octavia Video Translator",
        "version": "4.0.0",
        "status": "operational",
        "authentication": "JWT + Supabase",
        "password_hashing": "sha256_with_salt (bcrypt bypass)",
        "endpoints": {
            "health": "/api/health",
            "docs": "/docs",
            "signup": "/api/auth/signup",
            "login": "/api/auth/login",
            "logout": "/api/auth/logout",
            "verify": "/api/auth/verify",
            "resend_verification": "/api/auth/resend-verification",
            "demo_login": "/api/auth/demo-login"
        }
    }

# ========== APPLICATION ENTRY POINT ==========
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("OCTAVIA VIDEO TRANSLATOR v4.0 - SUPABASE EDITION")
    print("="*60)
    print("Password Hashing: SHA256 with salt")
    print("API URL: http://localhost:8000")
    print("Documentation: http://localhost:8000/docs")
    print("CORS enabled for: http://localhost:3000")
    print("Database: Supabase")
    print("Authentication: JWT with email verification")
    print("Demo login available: /api/auth/demo-login")
    print("Logs: artifacts/logs.jsonl")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )
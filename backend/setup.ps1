# Octavia Video Translator Setup Script for Windows
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "Octavia Video Translator Setup" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan

# Check Python
Write-Host "Checking Python..." -ForegroundColor Yellow
$pythonVersion = python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Please install Python 3.11+" -ForegroundColor Red
    exit 1
}
Write-Host "SUCCESS: $pythonVersion" -ForegroundColor Green

# Check FFmpeg
Write-Host "Checking FFmpeg..." -ForegroundColor Yellow
try {
    $null = ffmpeg -version 2>&1
    Write-Host "SUCCESS: FFmpeg detected" -ForegroundColor Green
} catch {
    Write-Host "WARNING: FFmpeg not found in PATH" -ForegroundColor Yellow
    Write-Host "Please install FFmpeg from: https://ffmpeg.org/download.html" -ForegroundColor Yellow
}

# Create directories
Write-Host "Creating directories..." -ForegroundColor Yellow
$directories = @("artifacts", "test_samples", "temp", "outputs", "models")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created: $dir" -ForegroundColor Cyan
    }
}

# Create virtual environment
Write-Host "Setting up virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "Created virtual environment" -ForegroundColor Cyan
} else {
    Write-Host "Virtual environment already exists" -ForegroundColor Cyan
}

# Activate venv
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv\Scripts\Activate.ps1") {
    .\venv\Scripts\Activate.ps1
}

# Install packages
Write-Host "Installing Python packages..." -ForegroundColor Yellow
pip install --upgrade pip
pip install fastapi uvicorn openai-whisper transformers torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install edge-tts ffmpeg-python pydub numpy
pip install supabase python-jose[cryptography] passlib[bcrypt] python-multipart python-dotenv

Write-Host ""
Write-Host "SETUP COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Put your test videos in 'test_samples' folder" -ForegroundColor Cyan
Write-Host "2. Create .env file with Supabase credentials" -ForegroundColor Cyan
Write-Host "3. Run: python run_simple.py --all" -ForegroundColor Cyan
Write-Host "4. Or run: python -m modules.pipeline --input test_samples/your_video.mp4 --target de" -ForegroundColor Cyan
Write-Host ""
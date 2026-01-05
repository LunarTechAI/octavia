"""
Octavia Backend - AI Video Translation Platform
Persistent Job Storage Implementation
"""
import os
import json
import uuid
import secrets
import logging
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from fastapi import FastAPI, Request, Response, HTTPException, Depends, Form, File, UploadFile, BackgroundTasks, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from shared_dependencies import (
    supabase, User, verify_password, get_password_hash,
    create_access_token, get_current_user, password_manager
)
from services.job_storage import job_storage
from config import HELSINKI_MODELS, DEMO_MODE, ENABLE_TEST_MODE
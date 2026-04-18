import os
import uuid
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import Dict, Any

from backend.auth import get_current_user
from backend.db import update_profile_picture, get_user_by_id, log_activity

router = APIRouter(prefix="/api/users", tags=["users"])

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "uploads"

@router.get("/me")
async def get_me(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Return the current active user profile."""
    safe_user = dict(current_user)
    safe_user.pop("password_hash", None)
    return safe_user

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp"}

@router.post("/me/picture")
async def upload_profile_picture(
    file: UploadFile = File(...), 
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Upload a profile picture for the current user."""
    if file.content_type not in ALLOWED_MIMES:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG, PNG, WEBP allowed.")
        
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 5MB.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    file_ext = file.filename.split('.')[-1] if '.' in file.filename else 'png'
    safe_filename = f"user_{current_user['id']}_{uuid.uuid4().hex[:8]}.{file_ext}"
    file_path = UPLOAD_DIR / safe_filename
    
    try:
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")
        
    url_path = f"/uploads/{safe_filename}"
    update_profile_picture(current_user["id"], url_path)
    log_activity(current_user["id"], "upload_profile_picture")
    
    return {"profile_picture_url": url_path}

@router.get("/{user_id}")
async def get_user_profile(user_id: int, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Allow any authenticated user to view basic profiles."""
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    return {
        "id": user["id"],
        "email": user["email"],
        "role": user["role"],
        "status": user["status"],
        "profile_picture_url": user["profile_picture_url"],
        "created_at": user["created_at"]
    }

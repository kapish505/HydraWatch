from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from typing import Dict, Any
from datetime import timedelta

from backend.auth import verify_password, create_access_token, get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES
from backend.db import get_user_by_email, log_activity

router = APIRouter(prefix="/api/auth", tags=["auth"])

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user_by_email(form_data.username)
    if not user or not verify_password(form_data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if user["status"] != "active":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is suspended")
        
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user["id"]), "role": user["role"]}, expires_delta=access_token_expires
    )
    
    # Log the login activity
    log_activity(user["id"], "login")
    
    return {"access_token": access_token, "token_type": "bearer", "user": {
        "id": user["id"], "email": user["email"], "role": user["role"], "profile_picture_url": user["profile_picture_url"]
    }}

@router.post("/logout")
async def logout(current_user: Dict[str, Any] = Depends(get_current_user)):
    log_activity(current_user["id"], "logout")
    return {"message": "Successfully logged out"}

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from pydantic import BaseModel

from backend.auth import get_current_active_admin
from backend.db import get_all_users, update_user_status, get_activity_logs, log_activity

router = APIRouter(prefix="/api/admin", tags=["admin"])

@router.get("/users")
async def list_users(limit: int = 50, offset: int = 0, current_admin: Dict[str, Any] = Depends(get_current_active_admin)):
    users = get_all_users(limit=limit, offset=offset)
    return {"users": users, "limit": limit, "offset": offset}

class StatusUpdate(BaseModel):
    status: str

@router.put("/users/{user_id}/status")
async def update_status(user_id: int, payload: StatusUpdate, current_admin: Dict[str, Any] = Depends(get_current_active_admin)):
    if payload.status not in ["active", "suspended"]:
        raise HTTPException(status_code=400, detail="Status must be 'active' or 'suspended'")
        
    if user_id == current_admin["id"]:
        raise HTTPException(status_code=400, detail="Cannot suspend your own admin account")
        
    success = update_user_status(user_id, payload.status)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
        
    log_activity(current_admin["id"], f"updated_user_{user_id}_status_to_{payload.status}")
    return {"message": f"User status updated to {payload.status}"}

@router.get("/activity")
async def list_activity(limit: int = 50, current_admin: Dict[str, Any] = Depends(get_current_active_admin)):
    logs = get_activity_logs(limit=limit)
    return {"logs": logs}

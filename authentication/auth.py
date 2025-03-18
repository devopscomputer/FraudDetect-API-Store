from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME)

async def authenticate(api_key: str = Depends(api_key_header)):
    if api_key != "YOUR_SECRET_API_KEY":  # Substitua por uma verificação real
        raise HTTPException(status_code=403, detail="Unauthorized")
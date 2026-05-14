from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

import httpx
from . import auth, users
from .schemas import Token, UserIn

client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(users.init_db, None)
    await run_in_threadpool(users.load_predefined_users, None, True)
    global client
    client = httpx.AsyncClient()
    yield
    await client.aclose()


app = FastAPI(title="MASB Auth Service", lifespan=lifespan)


def _get_client() -> httpx.AsyncClient:
    if client is None:
        raise HTTPException(status_code=503, detail="FHIR proxy client is not initialized")
    return client


async def fhir_request(
    method: str,
    fhir_api_request: str,
    payload: Optional[Dict[str, Any]] = None,
) -> httpx.Response:
    """Forward one FHIR HTTP request through the auth wrapper."""
    if not fhir_api_request:
        raise HTTPException(status_code=400, detail="FHIR API request is empty")

    http = _get_client()
    method = method.lower()
    if method == "get":
        response = await http.get(fhir_api_request)
    elif method == "put":
        response = await http.put(fhir_api_request, json=payload or {})
    elif method == "post":
        response = await http.post(fhir_api_request, json=payload or {})
    elif method == "delete":
        response = await http.delete(fhir_api_request)
    else:
        raise HTTPException(status_code=400, detail="Unsupported HTTP method")

    if not 200 <= response.status_code < 300:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"FHIR API request failed: {response.text}",
        )
    return response


def fhir_json_response(response: httpx.Response) -> JSONResponse:
    """Return proxied FHIR response content, including empty 204 responses."""
    if response.content:
        return JSONResponse(content=response.json(), status_code=response.status_code)
    return JSONResponse(content={}, status_code=response.status_code)


@app.post("/login", response_model=Token)
async def login(payload: UserIn):
    """Authenticate and return a JWT access token (3600s)."""
    user = await run_in_threadpool(users.get_user, payload.username)
    if not user:
        raise HTTPException(
            status_code=400, detail="Incorrect username or password")

    ok = await run_in_threadpool(users.verify_password, payload.password, user["hashed_password"])
    if not ok:
        raise HTTPException(
            status_code=400, detail="Incorrect username or password")

    role = user.get("role") if isinstance(user, dict) else None
    token = auth.create_access_token(data={"sub": payload.username, "role": role})
    return JSONResponse({
        "access_token": token,
        "token_type": "bearer",
        "expires_in": auth.ACCESS_TOKEN_EXPIRE_SECONDS,
        "role": role,
    })


@app.get("/list_users")
async def list_users(current_user: Dict = Depends(auth.get_administrator)):
    """List all users (administrator only)."""
    users_list = await run_in_threadpool(users.get_all_users)
    return {"users": users_list}


# Patient-specific endpoints
@app.put("/patient/update")
async def patient_update(
    fhir_api_request: str,  # FHIR HTTP RESTful API call, as a required query parameter
    payload: Dict,  # JSON body containing the FHIR API request details
    current_user: Dict = Depends(auth.get_patient)
):
    """Update(write) endpoint for patient role."""
    response = await fhir_request("put", fhir_api_request, payload)
    return fhir_json_response(response)


@app.post("/patient/create")
async def patient_create(
    fhir_api_request: str,
    payload: Dict,
    current_user: Dict = Depends(auth.get_patient)
):
    """Create endpoint for patient role."""
    response = await fhir_request("post", fhir_api_request, payload)
    return fhir_json_response(response)


@app.get("/patient/read")
async def patient_read(fhir_api_request: str, current_user: Dict = Depends(auth.get_patient)):
    """Read endpoint for patient role."""
    response = await fhir_request("get", fhir_api_request)
    return fhir_json_response(response)


# Physician-specific endpoints
@app.put("/physician/update")
async def physician_update(
    fhir_api_request: str,
    payload: Dict,
    current_user: Dict = Depends(auth.get_physician)
):
    """Update (write) endpoint for physician."""
    response = await fhir_request("put", fhir_api_request, payload)
    return fhir_json_response(response)


@app.post("/physician/create")
async def physician_create(
    fhir_api_request: str,
    payload: Dict,
    current_user: Dict = Depends(auth.get_physician)
):
    """Create endpoint for physician."""
    response = await fhir_request("post", fhir_api_request, payload)
    return fhir_json_response(response)


@app.get("/physician/read")
async def physician_read(fhir_api_request: str, current_user: Dict = Depends(auth.get_physician)):
    """Read endpoint for physician."""
    response = await fhir_request("get", fhir_api_request)
    return fhir_json_response(response)


# Administrator-specific endpoints
@app.put("/admin/update")
async def admin_update(
    fhir_api_request: str,
    payload: Dict,
    current_user: Dict = Depends(auth.get_administrator),
):
    """Update (write) endpoint for administrator."""
    response = await fhir_request("put", fhir_api_request, payload)
    return fhir_json_response(response)


@app.post("/admin/create")
async def admin_create(
    fhir_api_request: str,
    payload: Dict,
    current_user: Dict = Depends(auth.get_administrator)
):
    """Create endpoint for administrator."""
    response = await fhir_request("post", fhir_api_request, payload)
    return fhir_json_response(response)


@app.get("/admin/read")
async def admin_read(fhir_api_request: str, current_user: Dict = Depends(auth.get_administrator)):
    """Read endpoint for admin."""
    response = await fhir_request("get", fhir_api_request)
    return fhir_json_response(response)


@app.delete("/admin/delete")
async def admin_delete(fhir_api_request: str, current_user: Dict = Depends(auth.get_administrator)):
    """Delete endpoint for administrator."""
    response = await fhir_request("delete", fhir_api_request)
    return fhir_json_response(response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("masb_auth.main:app", host="0.0.0.0", port=8000, reload=True)

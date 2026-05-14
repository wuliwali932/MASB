````markdown
# MASB SMARTonFHIR Auth Service

Small FastAPI service providing a REST login endpoint and JWT-based access tokens.

Endpoints
- POST /login : JSON body {"username": "alice", "password": "secret"} -> returns access_token, token_type, expires_in
- GET /protected : Requires Authorization: Bearer <token>

Run locally

```bash
python -m uvicorn masb_auth.main:app --reload

source medagentbench/bin/activate 
```

Test login (example):

```bash
curl -X POST http://127.0.0.1:8000/login -H "Content-Type: application/json" -d '{"username": "patient1", "password":"p1secret"}'



```


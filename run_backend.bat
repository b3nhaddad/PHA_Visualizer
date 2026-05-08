@echo off
REM Start the FastAPI backend (local dev — binds to 127.0.0.1 to avoid Windows Firewall prompts).
REM API will be available at http://localhost:8000
REM Docs at http://localhost:8000/docs

cd backend
uvicorn api.main:app --host 127.0.0.1 --port 8000
